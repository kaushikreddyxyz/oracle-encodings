import subprocess,json,shutil,concurrent.futures
from pathlib import Path
r=Path(__file__).resolve().parent

def one(g):
 d=json.loads((r/f'connection-g{g}.json').read_text());ssh=d['ssh']
 subprocess.run(ssh+[f"pkill -f '^bash job.sh$' || true\npkill -f '^python -u code/experiment.py --run models/g{g}-' || true\nmv /workspace/output /workspace/output-initial-overlap\nmv /workspace/campaign.log /workspace/campaign-initial-overlap.log\nmkdir -p /workspace/output"],check=True)
 dest=r/'initial_overlap_runs'/f'g{g}';dest.parent.mkdir(exist_ok=True)
 if (r/'results'/f'g{g}').exists():shutil.move(str(r/'results'/f'g{g}'),str(dest))
 subprocess.run(['scp','-q','-i',str(Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-P',str(d['port']),str(r/'experiment.py'),str(r/'methods.py'),f"root@{d['ip']}:/workspace/code/"],check=True)
 subprocess.run(ssh+['cd /workspace\nnohup bash job.sh > campaign.log 2>&1 < /dev/null &\necho rerun-started'],check=True)
 print(g,'deduplicated rerun launched',flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(one,[5,6]))
