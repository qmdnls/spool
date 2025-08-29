import json
from pathlib import Path

import numpy as np

class Shard:
    def __init__(self, root: Path, shard_idx: int):
        self.dir = root / f"shard-{shard_idx:05d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bin = open(self.dir / "tokens.bin", "wb")
        self.sizes = []
        self.num_tokens = 0
        self.num_docs = 0

    def add(self, ids: np.ndarray):
        self.bin.write(ids.astype(np.uint32, copy=False).tobytes())
        self.sizes.append(int(ids.size))
        self.num_tokens += int(ids.size)
        self.num_docs += 1

    def close(self):
        self.bin.close()
        np.save(self.dir / "sizes.npy", np.asarray(self.sizes, dtype=np.int32))
        with open(self.dir / "meta.json", "w") as f:
            json.dump({"docs": self.num_docs, "tokens": self.num_tokens}, f)
