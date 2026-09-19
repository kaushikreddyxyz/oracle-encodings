import json,subprocess,tarfile,time,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent; RESULTS=ROOT.parent/'curves/results'
def cmd(a,**kw):return subprocess.run(a,check=True,**kw)
def deploy(g,pilot=False):
    d=json.loads((ROOT/f'pod-g{g}.json').read_text());pid=d['id']
    for _ in range(40):
        info=json.loads(subprocess.check_output(['runpodctl','pod','get',pid]));ssh=info.get('ssh',{})
        if ssh.get('ip') and ssh.get('port'):break
        time.sleep(3)
    ip,port=ssh['ip'],str(ssh['port']);key=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')
    conn=['ssh','-i',key,'-p',port,'-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=10',f'root@{ip}']
    (ROOT/f'connection-g{g}.json').write_text(json.dumps({'id':pid,'grid':g,'ip':ip,'port':int(port),'ssh':conn},indent=2))
    for _ in range(30):
        p=subprocess.run(conn+['echo ready'],capture_output=True)
        if p.returncode==0:break
        time.sleep(3)
    jobs=[]
    archive=Path('/tmp')/f'mzrecover-g{g}.tar'
    with tarfile.open(archive,'w') as t:
        for p in ROOT.glob('*.py'):t.add(p,arcname='code/'+p.name)
        t.add(RESULTS/f'distance/sets/g{g}.npz',arcname=f'sets/g{g}.npz')
        for seed in (0,1):
            for cond in ('fixed','trainable'):
                runs=sorted(RESULTS.glob(f'runpod-*/g{g}-s{seed}/fast-inj-{cond}-*'))
                runs=[r for r in runs if (r/'model.final.pt').exists()]
                run=runs[0];tag=f'g{g}-s{seed}-{cond}'
                jobs.append(tag)
                for name in ['model.final.pt','injector.final.pt','config_holder.json']:t.add(run/name,arcname=f'models/{tag}/{name}')
        if g==6:
            for seed in (0,1):
                run=next(r for r in sorted(RESULTS.glob(f'runpod-*/g6-s{seed}/fast-baseline*')) if (r/'model.final.pt').exists());tag=f'g6-s{seed}-baseline';jobs.append(tag)
                t.add(run/'model.final.pt',arcname=f'models/{tag}/model.final.pt')
    cmd(['scp','-q','-i',key,'-P',port,str(archive),f'root@{ip}:/workspace/input.tar'])
    cmd(conn+['cd /workspace && tar --no-same-owner --no-same-permissions -xf input.tar && python -m pip install -q --break-system-packages scipy && mkdir -p output'])
    script='#!/bin/bash\nset -e\ncd /workspace\nexport OMP_NUM_THREADS=4\n'
    if pilot:
        script+=f'python -u code/experiment.py --run models/{jobs[0]} --set sets/g{g}.npz --out output/pilot --count 1000 --steps 20 --n-val 2 --n-test 3\n'
    else:
        for phase in ['fit','eval']:
            for tag in jobs:script+=f'python -u code/experiment.py --run models/{tag} --set sets/g{g}.npz --out output/{tag} --phase {phase} --count 20000 --steps 2000 --n-val 20 --n-test 60\n'
    script+='touch /workspace/ALL_DONE\n'
    cmd(conn+['cat > /workspace/job.sh'],input=script.encode())
    cmd(conn+['cd /workspace\nnohup bash job.sh > campaign.log 2>&1 < /dev/null &\necho launched'],stdout=subprocess.DEVNULL)
    print(g,'launched',flush=True)
if __name__=='__main__':
    import sys
    if '--pilot' in sys.argv:deploy(5,True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p:list(p.map(deploy,range(5,11)))
