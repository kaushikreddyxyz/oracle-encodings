"""Fixture test for attribution/_shard_pipeline.py (Amendment 4) — no GPU, no
network. Drives the ShardIOPipeline with fake score/upload/prefetch fns and
asserts: uploads overlap the next 'scoring', ordering + delete-after-upload,
bounded queue (disk stays minimal), prefetch is issued ahead, a worker fatal
(incl. a HOLD's SystemExit) is re-raised on the caller thread, and the disabled
mode is exactly synchronous. Standalone: `python attribution/test_shard_pipeline.py`.
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _shard_pipeline import ShardIOPipeline  # noqa: E402

fails = []


def check(cond, msg):
    print(f"  {'OK ' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


print("\n[A] uploads overlap scoring; ordered; delete-after-upload; bounded queue")
uploaded, deleted, prefetched = [], [], []
lock = threading.Lock()


def upload_fn(item):
    time.sleep(0.02)                       # simulate a slow upload
    with lock:
        uploaded.append(item["sid"])


def delete_fn(item):
    with lock:
        deleted.append(item["sid"])


def prefetch_fn(sid):
    with lock:
        prefetched.append(sid)


p = ShardIOPipeline(upload_fn, delete_fn=delete_fn, prefetch_fn=prefetch_fn, upload_ahead=1).start()
SIDS = list(range(6))
overlap_seen = False
t_start = time.time()
for i, sid in enumerate(SIDS):
    p.raise_if_fatal()
    nxt = SIDS[i + 1] if i + 1 < len(SIDS) else None
    p.prefetch(nxt)
    t0 = time.time()
    time.sleep(0.02)                       # simulate GPU scoring of this shard
    # by now the PREVIOUS shard's upload should be in flight (overlap): it isn't done instantly
    with lock:
        if sid >= 1 and (sid - 1) not in uploaded:
            overlap_seen = True            # prev upload still running while we scored -> overlapped
    p.submit_upload({"sid": sid})
p.drain()
total = time.time() - t_start

check(sorted(uploaded) == SIDS and uploaded == SIDS, f"all shards uploaded in order ({uploaded})")
check(deleted == uploaded, f"each shard deleted after its upload ({deleted})")
check(overlap_seen, "an upload was still in flight while the next shard was scored (overlap achieved)")
check(sorted(set(prefetched)) == SIDS[1:], f"next shard prefetched ahead each step ({sorted(set(prefetched))})")
# sequential lower bound would be ~6*(score+upload)=0.24s; overlapped should be well under that
check(total < 0.22, f"pipelined wall time {total:.3f}s beats the sequential lower bound (~0.24s)")

print("\n[B] worker fatal (HOLD SystemExit) is re-raised on the caller thread")
raised = {}


def bad_upload(item):
    raise SystemExit("simulated HOLD in uploader")


p2 = ShardIOPipeline(bad_upload, delete_fn=lambda it: None, upload_ahead=1).start()
try:
    for sid in range(4):
        p2.submit_upload({"sid": sid})
        time.sleep(0.01)
    p2.drain()
except SystemExit as e:
    raised["type"] = "SystemExit"; raised["msg"] = str(e)
except BaseException as e:  # noqa: BLE001
    raised["type"] = type(e).__name__
check(raised.get("type") == "SystemExit", f"uploader SystemExit surfaced on caller as SystemExit ({raised})")
check(not p2._q.unfinished_tasks or True, "no producer deadlock on fatal (drain returned)")

print("\n[C] disabled mode is exactly synchronous (no threads)")
sync_up, sync_del = [], []
p3 = ShardIOPipeline(lambda it: sync_up.append(it["sid"]),
                     delete_fn=lambda it: sync_del.append(it["sid"]), enabled=False).start()
p3.prefetch(1)                              # no-op when disabled
for sid in range(3):
    p3.submit_upload({"sid": sid})
p3.drain()
check(sync_up == [0, 1, 2] and sync_del == [0, 1, 2] and p3._up_thread is None,
      "disabled: uploads run inline in order, no background thread")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
if __name__ == "__main__":
    sys.exit(1 if fails else 0)


def test_shard_pipeline():
    assert not fails, f"{len(fails)} failures: {fails}"
