import json,subprocess,tarfile,time,concurrent.futures,shlex
from pathlib import Path
r=Path(__file__).resolve().parent
JOBS={13:'g10-s0-trainable',14:'g10-s1-fixed',15:'g10-s1-trainable'}
def one(g):
 tag=JOBS[g];p=json.loads((r/f'pod-g{g}.json').read_text());info=json.loads(subprocess.check_output(['runpodctl','pod','get',p['id']]));s=info['ssh'];key=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')
 ssh=['ssh','-i',key,'-p',str(s['port']),'-o','StrictHostKeyChecking=accept-new',f"root@{s['ip']}"]
 (r/f'connection-g{g}.json').write_text(json.dumps({'id':p['id'],'grid':g,'job':tag,'ip':s['ip'],'port':s['port'],'ssh':ssh},indent=2))
 for _ in range(40):
  if subprocess.run(ssh+['echo ready'],capture_output=True).returncode==0:break
  time.sleep(2)
 manifest=json.loads((r/'input_manifest.json').read_text());source=next(m['source'] for m in manifest if tag==f"g{m['grid']}-s{m['seed']}-{m['condition']}")
 arc=Path(f'/tmp/mzrecover-helper{g}.tar')
 with tarfile.open(arc,'w') as t:
  for f in r.glob('*.py'):t.add(f,arcname='code/'+f.name)
  for f in (r/'results/g10'/tag).glob('*'):
   if f.name.endswith('-geometry.json') or f.name.endswith('-coef.pt') or f.name=='summary.json':t.add(f,arcname=f'output/{tag}/{f.name}')
  for name in ['model.final.pt','injector.final.pt']:t.add(Path(source)/name,arcname='model/'+name)
  t.add(r.parent/'curves/results/distance/sets/g10.npz',arcname='sets/g10.npz')
 subprocess.run(['scp','-q','-i',key,'-P',str(s['port']),str(arc),f"root@{s['ip']}:/workspace/input.tar"],check=True)
 subprocess.run(ssh+['cd /workspace && tar --no-same-owner --no-same-permissions -xf input.tar && python -m pip install -q --break-system-packages scipy'],check=True)
 script=f'#!/bin/bash\nset -e\ncd /workspace\npython -u code/experiment.py --run model --set sets/g10.npz --out output/{tag} --phase eval --n-val 20 --n-test 60\ntouch /workspace/ALL_DONE\n'
 subprocess.run(ssh+['cat > /workspace/job.sh'],input=script.encode(),check=True)
 subprocess.run(ssh+['cd /workspace\nnohup bash job.sh > campaign.log 2>&1 < /dev/null &\necho launched'],check=True)
 # Stream completed eval artifacts back to the original pod, whose queued model can skip them.
 target=json.loads((r/'connection-g10.json').read_text());dest=r/'results'/f'g{g}';dest.mkdir(exist_ok=True,parents=True)
 for _ in range(100):
  subprocess.run(['rsync','-az','-e',shlex.join(ssh[:-1]),ssh[-1]+':/workspace/output/',str(dest)+'/'],capture_output=True)
  src=dest/tag
  if src.exists() and not (r/'results/g10/TERMINATED.json').exists():
   subprocess.run(['rsync','-az','--include=*/','--include=*-eval.npz','--include=*.json','--exclude=*','-e',shlex.join(target['ssh'][:-1]),str(src)+'/',target['ssh'][-1]+f':/workspace/output/{tag}/'],capture_output=True)
  # Once DONE is synced, the monitor independently verifies and terminates this helper.
  if (src/'DONE').exists():break
  time.sleep(15)
 print(g,'helper results relayed',flush=True)
with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(one,JOBS))
