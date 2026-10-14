#!/usr/bin/env python3
"""Export a bounded, traceable real-data teaching collection in a CPU allocation.

No training, inference, downloads or dataset-wide search. Fixed three 2021 events;
inspect at most their first 20 dates. Keep source arrays read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

EVENTS = ('2021/fire_24935867.hdf5', '2021/fire_24935874.hdf5', '2021/fire_24935885.hdf5')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Data processing requires a Slurm CPU allocation.')
    if args.output.exists():
        raise SystemExit('Choose a new output directory.')
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from reproductions.wsts_fast_track.missingness import apply_corruption

    args.output.mkdir(parents=True)
    plt.rcParams.update({'font.size':11, 'axes.titlesize':12, 'figure.facecolor':'white', 'savefig.facecolor':'white'})
    fire_cmap = ListedColormap(['#edf2f7','#d94801'])
    error_cmap = ListedColormap(['#edf2f7','#0072b2','#d55e00','#cc79a7'])
    manifest = {'schema':1,'source_commit':os.environ.get('WILDFIRE_SOURCE_COMMIT'),
        'job_id':os.environ['SLURM_JOB_ID'], 'source_dataset':'WSTS original 2021 events in the corrected project HDF5 collection',
        'source_record':'https://zenodo.org/records/8006177',
        'selection':'Three fixed event IDs from the committed inventory; first consecutive three-date window with nonzero fire on last input and target, target indices 6..19. Selected for concept visibility, not representativeness or model performance.',
        'spatial_scope':'Center 128x128 source pixels, no smoothing or spatial reprojection. Row/column coordinates, not a basemap.',
        'model_scope':'No neural model inference. Persistence copies the previous observed binary mask unchanged.',
        'events':[]}

    def normalized(v):
        s = v.decode() if isinstance(v,bytes) else str(v)
        s = s.replace('.tif','')
        for form in ('%Y-%m-%d','%Y%m%d'):
            try:return datetime.strptime(s,form).date().isoformat()
            except ValueError:pass
        raise ValueError('Unrecognized date '+s)

    for relative in EVENTS:
        source=args.data/relative
        with h5py.File(source,'r') as f:
            d=f['data']; assert d.ndim==4 and d.shape[1]==23
            height,width=d.shape[-2:]; assert min(height,width)>=128
            top,left=int(round((height-128)/2)),int(round((width-128)/2))
            dates=[normalized(x) for x in d.attrs['img_dates']]
            selected=None
            for t in range(6,min(20,len(dates))):
                ds=[datetime.fromisoformat(x) for x in dates[t-2:t+1]]
                if any(b-a!=timedelta(days=1) for a,b in zip(ds,ds[1:])):continue
                fire=np.asarray(d[t-2:t+1,22,top:top+128,left:left+128])
                if np.any(fire[-2]>0) and np.any(fire[-1]>0):
                    selected=t;break
            if selected is None:raise ValueError('No qualifying window in fixed event '+relative)
            raw=np.asarray(d[t-1:t,:,top:top+128,left:left+128])
            shape=list(d.shape)
        ident=Path(relative).stem
        event_dir=args.output/ident;event_dir.mkdir()
        previous,current,target=fire>0
        error=np.zeros(target.shape,dtype=np.uint8)
        error[current & target]=1
        error[current & ~target]=2
        error[~current & target]=3
        counts={'tp':int((current&target).sum()),'fp':int((current&~target).sum()),'fn':int((~current&target).sum()),'tn':int((~current&~target).sum())}
        event={'id':ident,'source_file':relative,'source_shape':shape,'crop':[top,left,128,128],
            'dates':dates[t-2:t+1],'target_index':t,'source_bytes':source.stat().st_size,
            'raw_input_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
            'fire_window_sha256':hashlib.sha256(fire.tobytes()).hexdigest(),
            'positive_pixels':[int(x.sum()) for x in (previous,current,target)],
            'persistence_counts':counts,'figures':[]}
        def save(name,title,arrays,labels,cmaps=None,scales=None):
            n=len(arrays);cols=2 if n==4 else min(3,n);rows=(n+cols-1)//cols
            fig,axs=plt.subplots(rows,cols,figsize=(4.2*cols,4.2*rows),squeeze=False,layout='constrained')
            for i,ax in enumerate(axs.flat):
                if i>=n:ax.set_visible(False);continue
                a=arrays[i]; cmap=(cmaps or [fire_cmap]*n)[i]
                kw={'vmin':0,'vmax':1} if scales is None else scales[i]
                im=ax.imshow(a,cmap=cmap,interpolation='nearest',**kw)
                ax.set_title(labels[i]);ax.set_xlabel('Pixel column');ax.set_ylabel('Pixel row')
                if scales is not None and not isinstance(cmap, ListedColormap):
                    fig.colorbar(im,ax=ax,shrink=.75,label='Stored feature value')
            fig.suptitle(title+'\n'+ident+' | '+dates[t-1]+' → '+dates[t],fontsize=14)
            filename=name+'.png';fig.savefig(event_dir/filename,dpi=130);plt.close(fig)
            event['figures'].append({'name':name,'path':ident+'/'+filename,'title':title})
        save('timeline','Observed fire changes across dates',[previous,current,target],dates[t-2:t+1])
        maxfire=max(float(np.nanmax(fire)),1)
        save('fire-mask','A numeric active-fire channel becomes a binary mask',[fire[-2],current,target],['Stored active-fire channel','Input fire mask: value > 0','Next-day target: value > 0'],
             ['magma',fire_cmap,fire_cmap],[{'vmin':0,'vmax':maxfire},{'vmin':0,'vmax':1},{'vmin':0,'vmax':1}])
        channel_arrays=[raw[0,i] for i in (3,6,14)]
        channel_scales=[]
        for arr in channel_arrays:
            finite=arr[np.isfinite(arr)];assert finite.size
            lo,hi=np.percentile(finite,[2,98]);hi=max(float(hi),float(lo)+1e-6)
            channel_scales.append({'vmin':float(lo),'vmax':hi})
        save('channels','Different features describe the same patch',channel_arrays,['NDVI · raw channel 3','Wind speed · raw channel 6','Elevation · raw channel 14'],['viridis','cividis','terrain'],channel_scales)
        masks=[];fractions={}
        for scenario in ('M00','M01','M06','M07'):
            c=apply_corruption(raw,scenario_id=scenario,event_relative_path=relative,target_date=dates[t])
            mask=np.ones(target.shape,dtype=bool) if scenario=='M01' else (c.spatial_mask if c.spatial_mask is not None else np.zeros(target.shape,dtype=bool))
            fractions[scenario]=float(mask.mean())
            masks.append(np.ma.array(current,mask=mask))
            if scenario in ('M06','M07'):
                assert np.array_equal(c.values[:,14],raw[:,14],equal_nan=True)
        missing_cmap=ListedColormap(['#edf2f7','#d94801']);missing_cmap.set_bad('#6b7280')
        save('missingness','Gray means deliberately hidden, not observed background',masks,['M00 · complete','M01 · all fire history hidden','M06 · 25% dynamic block','M07 · 50% dynamic block'],[missing_cmap]*4)
        save('persistence','Yesterday\'s mask is a simple forecast to challenge',[current,target,error],['Persistence forecast = last mask','Observed next-day target','Blue: hit · orange: false alarm\npurple: missed fire'],[fire_cmap,fire_cmap,error_cmap],[{'vmin':0,'vmax':1},{'vmin':0,'vmax':1},{'vmin':0,'vmax':3}])
        assert fractions == {'M00':0.0,'M01':1.0,'M06':0.25,'M07':0.5}
        assert sum(counts.values())==128*128
        event['channel_color_limits'] = channel_scales
        event['hidden_fractions']=fractions
        event['feature_display']='Raw stored values; 2nd–98th percentile color limits per channel and event. No unit conversion; color saturation is display-only; incomparable color scales across features.'
        manifest['events'].append(event)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'events':len(manifest['events']),'figures':sum(len(x['figures']) for x in manifest['events']),'output':str(args.output)}))

if __name__=='__main__':main()
