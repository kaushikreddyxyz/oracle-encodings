import subprocess,json,datetime,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parent
until=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
def one(g):
 if (root/f'pod-g{g}.json').exists(): return {'grid':g,'existing':True}
 p=subprocess.run(['runpodctl','pod','create','--name',f'mzrecover-g{g}','--template-id','runpod-torch-v280','--gpu-id','NVIDIA A100-SXM4-80GB','--gpu-count','1','--cloud-type','SECURE','--container-disk-in-gb','20','--volume-in-gb','20','--ports','22/tcp','--terminate-after',until],capture_output=True,text=True)
 if p.returncode: return {'grid':g,'error':p.stderr+p.stdout}
 d=json.loads(p.stdout); d['grid']=g
 (root/f'pod-g{g}.json').write_text(json.dumps(d,indent=2)); return {'grid':g,'id':d.get('id'), 'costPerHr':d.get('costPerHr')}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
  for d in pool.map(one,range(5,11)): print(json.dumps(d),flush=True)
