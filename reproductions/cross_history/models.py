"""Three independent changes to the existing spatial/temporal forecasters."""
import importlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .data import ROOT


def initial_payload(history):
    run = (ROOT / 'runs/corrected-B3-S0-3K-21093266' if history == 1 else
           ROOT / 'archive/pre-t1-cleanup-2026-09-04/runs/corrected-B5-S0-3K-21144563')
    record = json.loads((run / 'completed.json').read_text())
    checkpoint = Path(record['checkpoint'])
    if history == 5:
        checkpoint = run / checkpoint.relative_to(ROOT / 'runs' / run.name)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return payload, str(checkpoint)


def make_base(history, hparams):
    name = 'SMPModel' if history == 1 else 'SMPTempModel'
    args = dict(hparams)
    args['encoder_weights'] = None
    if history == 5:
        args.pop('use_doy', None)
    return getattr(importlib.import_module('models.' + name), name)(**args)


class ContextTransport(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(channels*3,16,1), nn.SiLU(), nn.Conv2d(16,channels,1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x, mask):
        m = F.adaptive_avg_pool2d(mask, x.shape[-2:])
        valid = 1-m
        contexts = []
        for grid in (1,4):
            numerator = F.adaptive_avg_pool2d(x*valid, grid)
            denominator = F.adaptive_avg_pool2d(valid, grid)
            context = numerator / denominator.clamp_min(1e-5)
            contexts.append(F.interpolate(context, x.shape[-2:], mode='bilinear', align_corners=False))
        return x + m*self.residual(torch.cat([x,*contexts],dim=1))


class MissingnessExperts(nn.Module):
    """Late residual experts selected by the observed corruption regime."""
    def __init__(self, channels):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(channels+2,16,3,padding=1), nn.SiLU(),
        )
        self.fire = nn.Conv2d(16,1,1)
        self.spatial = nn.Conv2d(16,1,1)
        for head in (self.fire,self.spatial):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, decoded, spatial, fire_invalid):
        shape = decoded.shape[-2:]
        spatial = F.interpolate(spatial,shape,mode='nearest')
        fire_only = F.interpolate((fire_invalid-spatial).clamp(0,1),shape,mode='nearest')
        features = self.trunk(torch.cat((decoded,spatial,fire_only),dim=1))
        # The experts can correct the whole forecast, while the local mask tells
        # them where evidence was removed. Clean samples activate neither head.
        spatial_active = spatial.amax(dim=(-2,-1),keepdim=True)
        fire_active = fire_only.amax(dim=(-2,-1),keepdim=True)
        return spatial_active*self.spatial(features) + fire_active*self.fire(features)


class SpatialImpactFiLM(nn.Module):
    """Use observed context to modulate forecast impact beyond a spatial hole."""
    def __init__(self, channels):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Conv2d(channels+1,channels,1), nn.SiLU(),
            nn.Conv2d(channels,channels*2,1),
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, decoded, spatial):
        mask = F.interpolate(spatial,decoded.shape[-2:],mode='nearest')
        valid = 1-mask
        context = (decoded*valid).sum(dim=(-2,-1),keepdim=True)
        context = context/valid.sum(dim=(-2,-1),keepdim=True).clamp_min(1.)
        fraction = mask.mean(dim=(-2,-1),keepdim=True)
        scale,bias = self.condition(torch.cat((context,fraction),dim=1)).chunk(2,dim=1)
        active = mask.amax(dim=(-2,-1),keepdim=True)
        return decoded + active*(decoded*torch.tanh(scale)+bias)


class ForecastAwareDynamicInpainting(nn.Module):
    """Fill only missing dynamic fields from valid and static context."""
    def __init__(self, channels, dynamic):
        super().__init__()
        self.dynamic = tuple(dynamic)
        self.predictor = nn.Sequential(
            nn.Conv2d(2*channels+1,32,3,padding=1), nn.SiLU(),
            nn.Conv2d(32,32,3,padding=4,dilation=4), nn.SiLU(),
            nn.Conv2d(32,len(self.dynamic),1),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(self, x, spatial):
        batch,time,channels,height,width = x.shape
        flat = x.reshape(batch*time,channels,height,width)
        mask = spatial[:,None].expand(batch,time,1,height,width).reshape(
            batch*time,1,height,width)
        valid = 1-mask
        context = (flat*valid).sum(dim=(-2,-1),keepdim=True)
        context = context/valid.sum(dim=(-2,-1),keepdim=True).clamp_min(1.)
        residual = self.predictor(torch.cat((flat,context.expand_as(flat),mask),dim=1))
        filled = flat.clone()
        filled[:,self.dynamic] = filled[:,self.dynamic] + mask*residual
        return filled.reshape(batch,time,channels,height,width)


class NormalizedDiffusionInpainting(nn.Module):
    """Propagate observed dynamic fields into a hole over increasing scales."""
    def __init__(self, dynamic):
        super().__init__()
        self.dynamic = tuple(dynamic)

    def forward(self, x, spatial):
        batch,time,channels,height,width = x.shape
        flat = x.reshape(batch*time,channels,height,width)
        mask = spatial[:,None].expand(batch,time,1,height,width).reshape(
            batch*time,1,height,width)
        values = flat[:,self.dynamic]
        filled = values
        known = 1-mask
        # Successive normalized averages propagate boundary observations into
        # large holes; the accumulated radius covers the 50% test block.
        for radius in (1,2,4,8,16,32):
            kernel = 2*radius+1
            denominator = F.avg_pool2d(known,kernel,1,radius)
            proposal = F.avg_pool2d(filled*known,kernel,1,radius)
            proposal = proposal/denominator.clamp_min(1e-6)
            take = (1-known)*(denominator > 0).to(known.dtype)
            filled = filled*(1-take)+proposal*take
            known = torch.maximum(known,take)
        result = flat.clone()
        result[:,self.dynamic] = values*(1-mask)+filled*mask
        return result.reshape(batch,time,channels,height,width)


class DistanceReliabilityPrompt(nn.Module):
    """Encode depth inside a missing region as multi-scale channel prompts."""
    def __init__(self, channels):
        super().__init__()
        self.tokens = nn.ParameterList([nn.Parameter(torch.zeros(c)) for c in channels])

    @staticmethod
    def depth(mask):
        remaining = mask.clamp(0,1)
        depth = torch.zeros_like(remaining)
        for _ in range(32):
            depth = depth + remaining
            remaining = -F.max_pool2d(-remaining,3,1,1)
        return depth/depth.amax(dim=(-2,-1),keepdim=True).clamp_min(1.)

    def forward(self, features, spatial):
        depth = self.depth(spatial)
        prompted=[]
        for feature,token in zip(features,self.tokens):
            scale = F.interpolate(depth,feature.shape[-2:],mode='bilinear',align_corners=False)
            prompted.append(feature+scale*token[None,:,None,None])
        return prompted


class Forecaster(nn.Module):
    def __init__(self, base, history, method):
        super().__init__()
        self.base = base
        self.history = history
        self.method = method
        self.channels = 40 if history == 1 else 33
        if method in ('context','context_adapter','context_transition'):
            self.transport = nn.ModuleList([ContextTransport(c) for c in base.model.encoder.out_channels[1:]])
        if method in ('transition','transition_decoupled','context_transition'):
            decoder_channels = base.model.segmentation_head[0].in_channels
            self.transition = nn.Conv2d(decoder_channels,3,1)
            nn.init.zeros_(self.transition.weight)
            nn.init.zeros_(self.transition.bias)
            with torch.no_grad():
                self.transition.bias[0] = -3.
        if method == 'missingness_experts':
            decoder_channels = base.model.segmentation_head[0].in_channels
            self.missingness_experts = MissingnessExperts(decoder_channels)
        if method == 'spatial_impact_film':
            decoder_channels = base.model.segmentation_head[0].in_channels
            self.spatial_impact_film = SpatialImpactFiLM(decoder_channels)
        if method in ('dynamic_inpaint','dynamic_inpaint_reconstruct','block_specialist_dynamic'):
            columns = tuple(range(40)) if history == 1 else (
                0,1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,
                22,23,24,25,26,27,28,29,30,31,32,38,39)
            dynamic_source = set(range(12)) | {15} | set(range(33,38))
            dynamic = tuple(i for i,column in enumerate(columns) if column in dynamic_source)
            self.dynamic_inpaint = ForecastAwareDynamicInpainting(self.channels,dynamic)
        if method == 'normalized_inpaint':
            columns = tuple(range(40)) if history == 1 else (
                0,1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,
                22,23,24,25,26,27,28,29,30,31,32,38,39)
            dynamic_source = set(range(12)) | {15} | set(range(33,38))
            dynamic = tuple(i for i,column in enumerate(columns) if column in dynamic_source)
            self.normalized_inpaint = NormalizedDiffusionInpainting(dynamic)
        if method == 'distance_prompt':
            self.distance_prompt = DistanceReliabilityPrompt(base.model.encoder.out_channels[1:])

    def features(self, x):
        encoder = self.base.model.encoder
        if self.history == 1:
            return list(encoder(x[:,0]))
        per_time = [encoder(x[:,t]) for t in range(self.history)]
        positions = torch.arange(self.history,device=x.device)[None].expand(x.shape[0],-1)
        last, attention = self.base.ltae(torch.stack([f[-1] for f in per_time],dim=1), batch_positions=positions)
        skips = [self.base.temporal_aggregator(torch.stack([f[s] for f in per_time],dim=1),attn_mask=attention)
                 for s in range(1,len(per_time[0])-1)]
        return [per_time[0][0],*skips,last]

    def forward(self, packed, details=False):
        x = packed[:,:,:self.channels]
        spatial = packed[:,-1,self.channels:self.channels+1]
        fire_invalid = packed[:,-1,self.channels+1:self.channels+2]
        reconstruction = None
        if self.method in ('dynamic_inpaint','dynamic_inpaint_reconstruct','block_specialist_dynamic'):
            x = self.dynamic_inpaint(x,spatial)
            if self.method == 'dynamic_inpaint_reconstruct':
                reconstruction = x[:,:,self.dynamic_inpaint.dynamic]
        if self.method == 'normalized_inpaint':
            x = self.normalized_inpaint(x,spatial)
        features = self.features(x)
        if self.method == 'distance_prompt':
            features = [features[0],*self.distance_prompt(features[1:],spatial)]
        if self.method in ('context','context_adapter','context_transition'):
            features = [features[0], *[layer(f,spatial) for layer,f in zip(self.transport,features[1:])]]
        decoded = self.base.model.decoder(*features)
        if self.method == 'spatial_impact_film':
            decoded = self.spatial_impact_film(decoded,spatial)
        logits = self.base.model.segmentation_head(decoded)
        if self.method == 'missingness_experts':
            logits = logits + self.missingness_experts(decoded,spatial,fire_invalid)
        aux = None
        if self.method in ('transition','transition_decoupled','context_transition'):
            base_logits = logits
            state, survival_delta, new_delta = self.transition(decoded).split(1,dim=1)
            survival, new = logits + survival_delta, logits + new_delta
            observed = x[:,-1,-1:].clamp(0,1)
            occupancy = observed*(1-fire_invalid) + state.sigmoid()*fire_invalid
            log_occupied = occupancy.clamp_min(1e-30).log()
            log_empty = (1-occupancy).clamp_min(1e-30).log()
            log_positive = torch.logaddexp(log_occupied+F.logsigmoid(survival), log_empty+F.logsigmoid(new))
            log_negative = torch.logaddexp(log_occupied+F.logsigmoid(-survival), log_empty+F.logsigmoid(-new))
            logits = log_positive-log_negative
            aux = (state,survival,new)
            if self.method == 'transition_decoupled':
                aux_state, aux_survival_delta, aux_new_delta = self.transition(decoded.detach()).split(1,dim=1)
                aux = (aux_state,base_logits.detach()+aux_survival_delta,base_logits.detach()+aux_new_delta)
        if details:
            return logits, features[-3:], aux, reconstruction
        return logits

    def compute_loss(self, logits, target):
        return self.base.compute_loss(logits, target)


def feature_loss(student, teacher, mask, target, *, spatial_only=False):
    # Normalize across feature channels, emphasize missing regions and fire.
    fire_focus = F.max_pool2d(target[:,None].float(),9,1,4)
    weight = mask*(1+fire_focus) if spatial_only else mask+fire_focus
    losses = []
    for sf,tf in zip(student,teacher):
        w = F.adaptive_avg_pool2d(weight, sf.shape[-2:])
        distance = (F.normalize(sf,dim=1) - F.normalize(tf,dim=1)).square().sum(dim=1,keepdim=True)
        losses.append((distance*w).sum()/w.sum().clamp_min(1.))
    return torch.stack(losses).mean()


def transition_loss(aux, clean, target):
    state, survival, new = aux
    previous = clean[:,-1,-1:].clamp(0,1)
    y = target[:,None].float()
    state_loss = F.binary_cross_entropy_with_logits(state,previous,pos_weight=state.new_tensor(10.))
    # Separate conditional risks; no next-day label is consumed at inference.
    s_loss = F.binary_cross_entropy_with_logits(survival,y,reduction='none')
    n_loss = F.binary_cross_entropy_with_logits(new,y,reduction='none')
    conditional = (s_loss*previous).sum()/previous.sum().clamp_min(1.)
    conditional += (n_loss*(1-previous)).sum()/(1-previous).sum().clamp_min(1.)
    return .05*state_loss + .05*conditional
