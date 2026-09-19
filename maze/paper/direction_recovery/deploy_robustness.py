import json,subprocess,tarfile,time
from pathlib import Path
r=Path(__file__).resolve().parent;p=json.loads((r/'pod-g12.json').read_text());info=json.loads(subprocess.check_output(['runpodctl','pod','get',p['id']]));s=info['ssh'];key=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')
ssh=['ssh','-i',key,'-p',str(s['port']),'-o','StrictHostKeyChecking=accept-new',f"root@{s['ip']}"]
(r/'connection-g12.json').write_text(json.dumps({'id':p['id'],'grid':12,'ip':s['ip'],'port':s['port'],'ssh':ssh},indent=2))
for _ in range(30):
 if subprocess.run(ssh+['echo ready'],capture_output=True).returncode==0:break
 time.sleep(2)
with tarfile.open('/tmp/mzrecover-robustness.tar','w') as t:
 for f in r.glob('*.py'):t.add(f,arcname='code/'+f.name)
 for cond in ['fixed','trainable']:
  path=next(p for p in sorted((r.parent/'curves/results').glob(f'runpod-*/g10-s0/fast-inj-{cond}-*/injector.final.pt')))
  t.add(path,arcname=f'models/{cond}/injector.final.pt')
subprocess.run(['scp','-q','-i',key,'-P',str(s['port']),'/tmp/mzrecover-robustness.tar',f"root@{s['ip']}:/workspace/input.tar"],check=True)
subprocess.run(ssh+['cd /workspace && tar --no-same-owner --no-same-permissions -xf input.tar && python -m pip install -q --break-system-packages scipy'],check=True)
script='#!/bin/bash\nset -e\ncd /workspace\npython -u code/robustness.py --models models --out output/robustness\ntouch /workspace/ALL_DONE\n'
subprocess.run(ssh+['cat > /workspace/job.sh'],input=script.encode(),check=True)
subprocess.run(ssh+['cd /workspace\nnohup bash job.sh > campaign.log 2>&1 < /dev/null &\necho launched'],check=True)
