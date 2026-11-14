"""Download completed shards and terminate only after independent validation.

Validation is run by validate.py (or the local arrival validator). Handles the
shared g5/g6 pilot pod and never inspects or changes unrelated account pods.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import fleet

ROOT = Path(__file__).resolve().parent


def collect(grid):
    if (ROOT / 'results' / f'g{grid}' / 'BACKUP_VERIFIED.json').exists():
        return
    fleet.collect(grid)


def tick():
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {pool.submit(collect, g): g for g in range(5, 11)}
        for future in as_completed(jobs):
            try:
                future.result()
            except Exception as exc:
                print('COLLECT_ERROR', jobs[future], str(exc), flush=True)
    groups = {}
    for grid in range(5, 11):
        connection = json.loads((ROOT / f'connection-g{grid}.json').read_text())
        groups.setdefault(connection['id'], []).append(grid)
    done = []
    for pod_id, grids in groups.items():
        marker = ROOT / f'TERMINATED-{pod_id}.json'
        if marker.exists():
            done.append(pod_id)
            continue
        for grid in grids:
            validation = ROOT / f'validation-g{grid}.json'
            if not validation.exists() or json.loads(validation.read_text())['status'] != 'passed':
                break
            if not (ROOT / 'results' / f'g{grid}' / 'BACKUP_VERIFIED.json').exists():
                break
        else:
            result = subprocess.run(['runpodctl', 'pod', 'delete', pod_id],
                                    check=True, capture_output=True, text=True, timeout=30)
            response = json.loads(result.stdout)
            assert response.get('deleted') is True and response['id'] == pod_id
            marker.write_text(json.dumps({'id': pod_id, 'grids': grids,
                'terminated_at': datetime.now(timezone.utc).isoformat(),
                'reason': 'All assigned shards downloaded, SHA-256 verified, and independently validated.',
                'response': response}, indent=2))
            print('TERMINATED', pod_id, grids, flush=True)
            done.append(pod_id)
    print('STATUS', datetime.now(timezone.utc).isoformat(),
          f'{len(done)}/{len(groups)} working pods terminated', flush=True)
    return len(done) == len(groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        finished = tick()
        if finished or not args.watch:
            break
        time.sleep(45)
