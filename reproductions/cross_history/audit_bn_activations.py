from pathlib import Path
import json,torch
from torch import nn
from reproductions.cross_history.data import setup,evaluation_dataset
from reproductions.cross_history.run_three_directions import make_forecaster,BankForecaster,apply_bn_bank
setup();torch.set_num_threads(2)
root=Path('/project/6085198/kulbear/wildfire/runs');reports={}
for h,j in [(1,21787290),(5,21787291)]:
 s=torch.load(root/f'three-directions-{h}-bn-{j}/result/checkpoint.pt',map_location='cpu',weights_only=False)
 model=make_forecaster(h,s['hyper_parameters'],'control');model.load_state_dict(s['state_dict']);model.cuda().eval()
 dataset=evaluation_dataset(h,2021,'M06');packed=torch.stack([dataset[i][0] for i in range(4)]).cuda()
 records={};current={};hooks=[]
 def hook(name):
  def capture(module,inputs,output):current[name]=dict(input_max=float(inputs[0].abs().max()),output_max=float(output.abs().max()),output_rms=float(output.square().mean().sqrt()))
  return capture
 for name,m in model.named_modules():
  if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):hooks.append(m.register_forward_hook(hook(name)))
 with torch.no_grad():
  for variant,bank in [('original',None),('common',s['common']),('conditional',s['banks'][1])]:
   if bank is not None:apply_bn_bank(model,bank)
   current={};logits=model(packed);records[variant]=dict(logit_min=float(logits.min()),logit_max=float(logits.max()),layers=current)
  direct=logits.clone();routed=BankForecaster(model,s['banks']).eval()(packed)
  torch.testing.assert_close(direct,routed)
  records['direct_vs_routed_max_error']=float((direct-routed).abs().max())
 for hook_handle in hooks:hook_handle.remove()
 reports[str(h)]=records
 print(h,{k:{a:v for a,v in r.items() if a!='layers'} if isinstance(r,dict) else r for k,r in records.items()},flush=True)
 del model,packed
out=root/'three-directions-analysis';out.mkdir(exist_ok=True);(out/'bn-activation-audit.json').write_text(json.dumps(reports,indent=2)+'\n')
