import json,subprocess,tarfile
from pathlib import Path
r=Path(__file__).resolve().parent
p=json.loads((r/'pod-g11.json').read_text());info=json.loads(subprocess.check_output(['runpodctl','pod','get',p['id']]));s=info['ssh'];key=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')
ssh=['ssh','-i',key,'-p',str(s['port']),'-o','StrictHostKeyChecking=accept-new',f"root@{s['ip']}"]
(r/'connection-g11.json').write_text(json.dumps({'id':p['id'],'grid':11,'ip':s['ip'],'port':s['port'],'ssh':ssh},indent=2))
with tarfile.open('/tmp/mzrecover-legacy.tar','w') as t:
 for f in r.glob('*.py'):t.add(f,arcname='code/'+f.name)
 for f in (r/'reference').glob('*.pt'):t.add(f,arcname='reference/'+f.name)
 t.add(r.parent/'curves/results/distance/sets/g6.npz',arcname='sets/g6.npz')
 for seed in [0,1]:
  path=next(p for p in sorted((r.parent/'curves/results').glob(f'runpod-*/g6-s{seed}/fast-baseline*/model.final.pt')))
  t.add(path,arcname=f'models/baseline-s{seed}/model.final.pt')
subprocess.run(['scp','-q','-i',key,'-P',str(s['port']),'/tmp/mzrecover-legacy.tar',f"root@{s['ip']}:/workspace/input.tar"],check=True)
subprocess.run(ssh+['cd /workspace && tar --no-same-owner --no-same-permissions -xf input.tar && python -m pip install -q --break-system-packages scipy'],check=True)
script='#!/bin/bash\nset -e\ncd /workspace\n'
for seed in [0,1]:script+=f'python -u code/legacy.py --run models/baseline-s{seed} --set sets/g6.npz --reference reference --out output/legacy-s{seed} --steps 2000\n'
script+='touch /workspace/ALL_DONE\n'
subprocess.run(ssh+['cat > /workspace/job.sh'],input=script.encode(),check=True)
subprocess.run(ssh+['cd /workspace\nnohup bash job.sh > campaign.log 2>&1 < /dev/null &\necho launched'],check=True)
