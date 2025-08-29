import time

class Heartbeat:
    def __init__(self, every_docs=100_000, every_seconds=30, label="encode"):
        self.every_docs = int(every_docs)
        self.every_seconds = float(every_seconds)
        self.label = label
        self.start = self.last = time.perf_counter()
        self.docs = 0
        self.toks = 0
        self._next = self.every_docs

    def update(self, n_docs=1, n_toks=0, shard_idx=None, shard_docs=None, shard_toks=None):
        self.docs += int(n_docs)
        self.toks += int(n_toks)
        now = time.perf_counter()
        if self.docs >= self._next or (now - self.last) >= self.every_seconds:
            dt = max(now - self.start, 1e-9)
            msg = (
                f"[{self.label}] docs={self.docs:,} toks={self.toks:,} "
                f"dps={self.docs/dt:,.0f} tps={self.toks/dt:,.0f}"
            )
            if shard_idx is not None:
                msg += f" shard={shard_idx:05d}"
                if shard_docs is not None:
                    msg += f" sdocs={shard_docs:,}"
                if shard_toks is not None:
                    msg += f" stoks={shard_toks:,}"
            print(msg, flush=True)
            self.last = now
            while self.docs >= self._next:
                self._next += self.every_docs

    def finish(self):
        # call once at the end
        self.update(0, 0)  # force a final timestamped line if interval hit
        dt = max(time.perf_counter() - self.start, 1e-9)
        print(
            f"[{self.label} done] docs={self.docs:,} toks={self.toks:,} "
            f"secs={dt:,.1f} dps={self.docs/dt:,.0f} tps={self.toks/dt:,.0f}",
            flush=True,
        )
