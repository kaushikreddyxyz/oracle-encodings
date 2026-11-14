import subprocess,json,time,concurrent.futures,argparse,hashlib,tarfile,shlex
from pathlib import Path
R=Path(__file__).resolve().parent;KEY=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')
def run(args,**kw):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=180,**kw).stdout
def deploy(g):
 p=R/f'pod-g{g}.json'
 if p.exists():d=json.loads(p.read_text())
 else:
  d=json.loads(run(['runpodctl','pod','create','--name',f'mzalign-g{g}','--template-id','runpod-torch-v280','--gpu-id','NVIDIA A100-SXM4-80GB','--gpu-count','1','--cloud-type','SECURE','--container-disk-in-gb','20','--volume-in-gb','20','--ports','22/tcp','--terminate-after','2026-09-21T22:00:00Z']))
  d={k:v for k,v in d.items() if k!='env'};d['grid']=g;p.write_text(json.dumps(d,indent=2));print('CREATED',g,d['id'],flush=True)
 for _ in range(100):
  info=json.loads(run(['runpodctl','pod','get',d['id']]));s=info.get('ssh',{})
  if s.get('ip'):break
  time.sleep(3)
 ssh=['ssh','-i',KEY,'-p',str(s['port']),'-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=15',f"root@{s['ip']}"]
 (R/f'connection-g{g}.json').write_text(json.dumps({'id':d['id'],'grid':g,'ip':s['ip'],'port':s['port'],'ssh':ssh},indent=2))
 if g==5 or (R/f'launched-g{g}').exists():return
 run(['scp','-q','-i',KEY,'-o','StrictHostKeyChecking=accept-new','-P',str(s['port']),'/tmp/mzalign-input.tar',f"root@{s['ip']}:/workspace/input.tar"])
 run(ssh+['cd /workspace && tar --no-same-owner -xf input.tar && python -m pip install -q --break-system-packages scipy'])
 run(ssh+[f'cd /workspace\nnohup flock -n /workspace/train.lock python -u sweep.py --grid {g} > campaign.log 2>&1 < /dev/null &\necho launched'])
 (R/f'launched-g{g}').write_text('launched');print('LAUNCHED',g,flush=True)
def collect(g):
 c=json.loads((R/f'connection-g{g}.json').read_text());ssh=c['ssh']
 workdir=c.get('workdir','/workspace');wd=shlex.quote(workdir)
 state=run(ssh+[f'cd {wd} && (test -f output/DONE && echo DONE || tail -2 campaign.log)'])
 if state.strip()!='DONE':print(g,state.strip(),flush=True);return
 dest=R/'results'/f'g{g}';dest.mkdir(parents=True,exist_ok=True)
 if not (dest/'BACKUP_VERIFIED.json').exists():
  remote=run(ssh+[f'cd {wd} && tar -czf results.tar.gz output campaign.log && sha256sum results.tar.gz']).split()[0]
  archive=dest/'results.tar.gz';run(['scp','-q','-i',KEY,'-P',str(c['port']),f"root@{c['ip']}:{workdir}/results.tar.gz",str(archive)])
  actual=hashlib.sha256(archive.read_bytes()).hexdigest();assert actual==remote
  with tarfile.open(archive) as tar:tar.extractall(dest,filter='data')
  (dest/'BACKUP_VERIFIED.json').write_text(json.dumps({'sha256':actual,'id':c['id']}))
 print('VERIFIED',g,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['deploy','collect']);a=p.parse_args()
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  jobs={pool.submit(deploy if a.mode=='deploy' else collect,g):g for g in range(5,11)}
  for f in concurrent.futures.as_completed(jobs):
   try:f.result()
   except Exception as e:print('ERROR',jobs[f],repr(e),getattr(e,'stderr',''),flush=True)
