from pathlib import Path
import json,torch
root=Path('/project/6085198/kulbear/wildfire/runs')
reports={}
for h,j in [(1,21787290),(5,21787291)]:
 p=root/f'three-directions-{h}-bn-{j}/result/checkpoint.pt'
 if not p.exists():continue
 s=torch.load(p,map_location='cpu',weights_only=False);rows=[]
 for name,c in s['common'].items():
  a,b=s['banks'][0][name],s['banks'][1][name]
  n=a['count']+b['count'];assert n==c['count']
  m=(a['mean'].double()*a['count']+b['mean'].double()*b['count'])/n
  v=((a['count']-1)*a['var'].double()+(b['count']-1)*b['var'].double()+a['count']*(a['mean'].double()-m)**2+b['count']*(b['mean'].double()-m)**2)/(n-1)
  orig=s['state_dict'][name+'.running_var']
  row=dict(layer=name,common_mean_merge_max=float((m-c['mean']).abs().max()),common_var_merge_max=float((v-c['var']).abs().max()),common_var_max=float(c['var'].max()),original_min_var=float(orig.min()))
  for tag,bank in [('absent',a),('present',b)]:
   row[tag]=dict(min_var=float(bank['var'].min()),zero_variance=int((bank['var']==0).sum()),channels=bank['var'].numel(),max_gain_ratio=float(((orig+1e-5)/(bank['var']+1e-5)).sqrt().max()))
  rows.append(row)
 reports[str(h)]=dict(samples=s['bank_samples'],layers=rows)
 print(h,'layers',len(rows),'maxmeanmergeerror',max(r['common_mean_merge_max'] for r in rows),'maxvarmergeerror',max(r['common_var_merge_max'] for r in rows))
 print(json.dumps(sorted(rows,key=lambda r:r['present']['max_gain_ratio'],reverse=True)[:5],indent=2))
out=root/'three-directions-analysis';out.mkdir(exist_ok=True)
(out/'bn-statistics-audit.json').write_text(json.dumps(reports,indent=2)+'\n')
