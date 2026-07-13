"""Overlap the per-shard I/O of score_climbmix_stacked.py with GPU scoring
(Amendment 4). The stock loop is strictly sequential per shard:

    download parquet  ->  GPU score  ->  upload  ->  delete

so the next shard's parquet download never overlaps the current shard's scoring,
and the current shard's upload never overlaps the next shard's scoring. This
module pipelines both around the (unchanged) main-thread GPU scoring:

  * ``prefetch(sid)`` warms the NEXT shard's parquet on a background thread while
    the current shard is scored (the parquet download has no watchdog today —
    moving it off-thread loses none);
  * ``submit_upload(item)`` hands a finished shard to a background uploader
    (bounded queue, default depth 1 => at most one shard's outputs pending, disk
    high-water ~3 shards: scoring + queued + uploading) so the upload overlaps
    the next shard's scoring; the uploader
    deletes the shard's files after a verified upload.

Watchdog semantics are preserved: ``with_retries`` in score_climbmix_stacked is
made thread-aware (SIGALRM on the main thread as before; an equivalent daemon-
thread timeout in worker threads, since SIGALRM only fires on the main thread —
same auto-recover-from-silent-hang behavior). A fatal error in a worker (incl. a
HOLD's SystemExit) is stashed and re-raised on the CALLER's thread via
``raise_if_fatal``/``drain`` so the main thread performs the HOLD/exit.

No GPU or network here — fixture-tested with fake score/upload/prefetch fns.
"""
import queue
import threading


class ShardIOPipeline:
    def __init__(self, upload_fn, *, delete_fn=None, prefetch_fn=None, log=print,
                 upload_ahead=1, enabled=True):
        self.upload_fn = upload_fn
        self.delete_fn = delete_fn
        self.prefetch_fn = prefetch_fn
        self.log = log
        self.enabled = enabled
        self._q = queue.Queue(maxsize=max(1, int(upload_ahead)))
        self._fatal = None
        self._up_thread = None
        self._pf_thread = None
        self._pf_q = queue.Queue()
        self._pf_seen = set()

    def start(self):
        if self.enabled:
            self._up_thread = threading.Thread(target=self._upload_loop, name="shard-upload", daemon=True)
            self._up_thread.start()
            if self.prefetch_fn is not None:
                self._pf_thread = threading.Thread(target=self._prefetch_loop, name="shard-prefetch", daemon=True)
                self._pf_thread.start()
        return self

    # -- background uploader --
    def _upload_loop(self):
        draining = False  # after a fatal, keep consuming so producers never block / join never hangs
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            if draining:
                self._q.task_done()
                continue
            try:
                self.upload_fn(item)
                if self.delete_fn is not None:
                    self.delete_fn(item)
            except BaseException as e:  # incl. SystemExit from a HOLD; surfaced on the caller thread
                self._fatal = e
                draining = True
            self._q.task_done()

    # -- background parquet prefetch (best-effort; failures are non-fatal) --
    def _prefetch_loop(self):
        while True:
            sid = self._pf_q.get()
            if sid is None:
                return
            try:
                self.prefetch_fn(sid)
            except Exception as e:  # noqa: BLE001 — prefetch is a cache-warm; main-thread download retries
                self.log(f"[pipeline] prefetch shard {sid} failed (non-fatal, will download inline): {e!r}")

    def prefetch(self, sid):
        if not self.enabled or self.prefetch_fn is None or sid is None or sid in self._pf_seen:
            return
        self._pf_seen.add(sid)
        self._pf_q.put(sid)

    def raise_if_fatal(self):
        if self._fatal is not None:
            raise self._fatal

    def submit_upload(self, item):
        """Enqueue a finished shard for background upload (blocks only if the
        bounded queue is full — backpressure keeps disk minimal). Synchronous when
        disabled. Re-raises any stashed uploader fatal on the caller's thread."""
        self.raise_if_fatal()
        if not self.enabled:
            self.upload_fn(item)
            if self.delete_fn is not None:
                self.delete_fn(item)
            return
        self._q.put(item)
        self.raise_if_fatal()

    def drain(self):
        """Wait for all pending uploads; re-raise a stashed fatal on the caller."""
        if self.enabled:
            self._q.join()
            self.raise_if_fatal()
            self._q.put(None)
            if self._pf_thread is not None:
                self._pf_q.put(None)
