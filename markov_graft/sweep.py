"""Run the (table x size x arm x seed) grid as P concurrent processes on one GPU.
Cells that already have results/<name>.json are skipped, so the sweep is resumable.
Cells are ordered most-informative-first (size m, seed 0, dense tables) so partial
results are already interpretable."""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

DENSE = [(k, None) for k in range(1, 8)]              # (k, m): A^k tables, ell = 8^k * 15.3 bits
SPARSE = [(7, 1), (7, 3), (7, 5)]                     # nominal order 7, table of size 8^m
ARMS = ["none", "oracle_1", "oracle_3", "leak_1"]


def cells(a):
    sizes = a.sizes.split(",")
    seeds = [int(x) for x in a.seed_list.split(",")] if a.seed_list else list(range(a.seeds))
    tables = DENSE + SPARSE
    if a.tables:
        want = [tuple(int(v) for v in t.split(":")) for t in a.tables.split(",")]
        tables = [(k, m) for (k, m) in tables if (k, k if m is None else m) in want]
    order = []
    for seed in seeds:
        for size in sizes:
            for (k, m) in tables:
                for arm in ARMS:
                    order.append((k, m, size, arm, seed))
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="m,s,l")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed-list", default="", help="explicit seeds, e.g. 1 or 0,1 (overrides --seeds)")
    ap.add_argument("--tables", default="", help="restrict to tables given as k:m, e.g. 7:7,7:1,7:3,7:5")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--extra", default="")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    todo = []
    for (k, m, size, arm, seed) in cells(a):
        mm = k if m is None else m
        name = f"A8_h2_k{k}_m{mm}_{size}_{arm}_s{seed}"
        if os.path.exists(os.path.join(HERE, "results", name + ".json")):
            continue
        cmd = [sys.executable, os.path.join(HERE, "train.py"), "--k", str(k), "--size", size, "--arm", arm,
               "--seed", str(seed), "--steps", str(a.steps)] + (["--m", str(m)] if m else []) + a.extra.split()
        todo.append((name, cmd))
    print(f"{len(todo)} cells to run", flush=True)
    if a.dry:
        for n, _ in todo: print(n)
        return
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    running = []
    while todo or running:
        while todo and len(running) < a.parallel:
            name, cmd = todo.pop(0)
            log = open(os.path.join(HERE, "logs", name + ".log"), "w")
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            print(time.strftime("%H:%M:%S"), "start", name, flush=True)
        time.sleep(5)
        still = []
        for name, p, log in running:
            if p.poll() is None:
                still.append((name, p, log))
            else:
                log.close()
                print(time.strftime("%H:%M:%S"), "done", name, "rc", p.returncode, flush=True)
        running = still


if __name__ == "__main__":
    main()
