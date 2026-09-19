"""Read-only progress and incremental result sync; terminate completed campaign pods after verification."""
import json,subprocess,time,concurrent.futures,hashlib,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def poll(p):
    d=json.loads(p.read_text());g=d['grid'];dest=ROOT/'results'/f'g{g}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'TERMINATED.json').exists():return f'g{g}: terminated'
    ssh=d['ssh'];command='test -f /workspace/ALL_DONE && echo ALL_DONE; tail -2 /workspace/campaign.log; command -v rsync'
    r=subprocess.run(ssh+[command],capture_output=True,text=True,timeout=30)
    (dest/'status.txt').write_text(r.stdout+r.stderr)
    if r.returncode and not r.stdout:return f'g{g}: {r.stderr[:100]}'
    complete=r.stdout.startswith('ALL_DONE\n')
    if '/usr/bin/rsync' not in r.stdout:
        subprocess.run(ssh+['apt-get update -qq && apt-get install -y -qq rsync'],capture_output=True,timeout=120,check=True)
    transport=shlex.join(ssh[:-1]);source=ssh[-1]+':/workspace/output/'
    args=['rsync','-az','--exclude=pilot/','-e',transport,source,str(dest)+'/']
    if complete:args.insert(2,'--checksum')
    subprocess.run(args,capture_output=True,timeout=240,check=True)
    log=subprocess.run(ssh+['cat /workspace/campaign.log'],capture_output=True,text=True,timeout=30)
    (dest/'campaign.log').write_text(log.stdout)
    if complete:
        # Verify all result/checkpoint files by SHA-256 before irreversibly terminating the pod.
        code="import os,hashlib,json; root='/workspace/output'; out={};\nfor dp,ds,fs in os.walk(root):\n ds[:]=[d for d in ds if d!='pilot']\n for f in fs:\n  p=os.path.join(dp,f);out[os.path.relpath(p,root)]=hashlib.sha256(open(p,'rb').read()).hexdigest()\nprint(json.dumps(out))"
        r=subprocess.run(ssh+['python -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=120)
        manifest=json.loads(r.stdout)
        assert manifest and any(k.endswith('/DONE') for k in manifest),'no completed runs'
        for name,sha in manifest.items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==sha,name
        (dest/'verified-manifest.json').write_text(json.dumps(manifest,indent=2))
        term=subprocess.run(['runpodctl','pod','delete',d['id']],capture_output=True,text=True)
        if term.returncode:
            (dest/'teardown-error.txt').write_text(term.stdout+term.stderr)
            return f'g{g}: verified, delete failed {term.stderr[:120]}'
        (dest/'TERMINATED.json').write_text(json.dumps({'id':d['id'],'time':time.time(),'verified_files':len(manifest),'response':term.stdout},indent=2))
        return f'g{g}: VERIFIED AND TERMINATED ({len(manifest)} files)'
    return f'g{g}: '+r.stdout[-250:].replace('\n',' | ')

if __name__=='__main__':
    import sys
    while True:
        ps=sorted(ROOT.glob('connection-g*.json'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            futures={pool.submit(poll,p):p for p in ps}
            for f in concurrent.futures.as_completed(futures):
                try:print(f.result(),flush=True)
                except Exception as e:print(futures[f].name,type(e).__name__,str(e)[:200],flush=True)
        if '--once' in sys.argv:break
        if len(ps)>=7 and all((ROOT/'results'/f"g{json.loads(p.read_text())['grid']}"/'TERMINATED.json').exists() for p in ps):break
        time.sleep(45)
