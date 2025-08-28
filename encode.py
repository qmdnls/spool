import argparse
from functools import partial

import numpy as np
import pandas as pd
import tiktoken

def parse_args():
    parser = argparse.ArgumentParser(
        prog="spool encoder",
        description="Encodes inputs offline for fast and efficient dataloading",
    )
    parser.add_argument("--input", type=str)
    parser.add_argument("--out-prefix", type=str, default="./data/")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    return parser.parse_args()

def get_tokenizer(name="gpt2"):
    return tiktoken.get_encoding(name)

def encode(tokenizer, text):
    ids = tokenizer.encode_ordinary(text)
    return ids

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_parquet(
        args.input,
        engine="pyarrow",
    )
    print(df)
    tokenizer = get_tokenizer(args.tokenizer)
    encode_fn = partial(encode, tokenizer)
    ids = df.content.apply(encode_fn)
    print(ids)
