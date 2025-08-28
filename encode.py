import pandas as pd
import tiktoken

if __name__ == "__main__":
    path = "shakespeare.parquet"
    df = pd.read_parquet(
        path,
        engine="pyarrow",
    )
    print(df)
