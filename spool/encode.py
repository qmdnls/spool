import argparse
import json
import os
from pathlib import Path
from typing import Union 

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import tiktoken

from spool.shard import Shard
from spool.logging import Heartbeat

def parse_args():
    parser = argparse.ArgumentParser(
        prog="spool encoder",
        description="Encodes inputs offline for fast and efficient dataloading",
    )
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    return parser.parse_args()

def get_tokenizer(name: str = "gpt2") -> tiktoken.core.Encoding:
    return tiktoken.get_encoding(name)

def iter_text_batches(input_path: str, batch_rows=1024, input_format="parquet"):
    dataset = ds.dataset(input_path, format=input_format)
    scanner = dataset.scanner(columns=["content"], batch_size=batch_rows)
    for rb in scanner.to_batches():
        col = rb.column(0).to_pylist()  # list[str|None]
        # drop nulls/empties here if you want
        texts = [t for t in col if t]    # keep deterministic order
        if texts:
            yield texts

def encode_and_shard(
    tokenizer: str,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    input_format: str = "parquet",
    max_tokens_per_shard: int = 1e7
) -> int:
    shard_stats = {}
    
    input_dir = Path(args.input).expanduser().resolve(strict=False)
    output_dir = Path(args.output).expanduser().resolve(strict=False)
    hb = Heartbeat(every_docs=100_000, every_seconds=30, label="encode")

    # load tokenizer
    tokenizer = get_tokenizer(args.tokenizer)
    
    # output shard we will write to
    shard_idx = 0
    current = Shard(output_dir, shard_idx)

    # iterate over all input parquets in batches with pyarrow dataset
    for texts in iter_text_batches(input_dir, batch_rows=8192, input_format=input_format):
        encoded = tokenizer.encode_ordinary_batch(texts)
        for ids in encoded:
            n = len(ids)
            
            # start new shard if adding this would exceed the max size 
            if current.num_tokens + n > max_tokens_per_shard and current.num_docs > 0:
                shard_stats[shard_idx] = {
                    "docs": current.num_docs,
                    "tokens": current.num_tokens,
                }
                current.close()
                shard_idx += 1
                current = ShardWriter(output_dir, shard_idx)
            
            # write to disk 
            current.add(np.asarray(ids, dtype=np.uint32))

            # update heartbeat
            hb.update(
                n_docs=1,
                n_toks=n,
                shard_idx=shard_idx,
                shard_docs=current.num_docs,
                shard_toks=current.num_tokens,
            )
       
    # close last shard
    shard_stats[shard_idx] = {
        "docs": current.num_docs,
        "tokens": current.num_tokens,
    } 
    current.close()

    # write manifest
    total_docs = int(sum(s["docs"] for s in shard_stats.values()))
    total_tokens = int(sum(s["tokens"] for s in shard_stats.values()))
    manifest = {
        "version": 1,
        "encoding": args.tokenizer,
        "num_shards": len(shard_stats),
        "docs": total_docs,
        "tokens": total_tokens,
        "shards": [{"path": f"shard-{i:05d}", **shard_stats[i]} for i in range(len(shard_stats))]
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
   
    return total_docs, total_tokens

if __name__ == "__main__":
    args = parse_args()
    total_docs, total_tokens = encode_and_shard(
        args.tokenizer,
        args.input,
        args.output,
    )
    print(f"Finished writing encoded data to `{args.output}`.")
    print(f"Total documents: {total_docs}")
    print(f"Total tokens: {total_tokens}")
