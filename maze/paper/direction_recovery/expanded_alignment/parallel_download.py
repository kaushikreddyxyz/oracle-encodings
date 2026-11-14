"""Download a completed archive over parallel SSH byte ranges, then verify."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent


def main(grid, streams):
    c = json.loads((ROOT / f'connection-g{grid}.json').read_text())
    remote = c.get('workdir', '/workspace') + '/results.tar.gz'
    command = f'stat -c %s {shlex.quote(remote)}; sha256sum {shlex.quote(remote)}'
    info = subprocess.run(c['ssh'] + [command], check=True, capture_output=True,
                          text=True, timeout=30).stdout.split()
    size, expected = int(info[0]), info[1]
    dest = ROOT / 'results' / f'g{grid}'
    parts = dest / 'transfer_parts'
    parts.mkdir(exist_ok=True)
    width = (size + streams - 1) // streams

    def get(i):
        start = i * width
        count = min(width, size - start)
        part = parts / f'{i:02d}'
        cmd = (f'dd if={shlex.quote(remote)} iflag=skip_bytes,count_bytes '
               f'skip={start} count={count} status=none')
        with part.open('wb') as f:
            subprocess.run(c['ssh'] + [cmd], stdout=f, check=True, timeout=600)
        assert part.stat().st_size == count
        print('PART', i, count, flush=True)

    with ThreadPoolExecutor(max_workers=streams) as pool:
        for future in as_completed([pool.submit(get, i) for i in range(streams)]):
            future.result()
    archive = dest / 'results.tar.gz'
    temp = dest / 'results.complete.tmp'
    digest = hashlib.sha256()
    with temp.open('wb') as out:
        for i in range(streams):
            with (parts / f'{i:02d}').open('rb') as f:
                while data := f.read(1024 * 1024):
                    digest.update(data)
                    out.write(data)
    assert temp.stat().st_size == size and digest.hexdigest() == expected
    temp.replace(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter='data')
    (dest / 'BACKUP_VERIFIED.json').write_text(json.dumps(
        {'sha256': expected, 'id': c['id'], 'bytes': size, 'transfer_streams': streams}, indent=2))
    print('VERIFIED', grid, size, expected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, required=True)
    parser.add_argument('--streams', type=int, default=8)
    args = parser.parse_args()
    main(args.grid, args.streams)
