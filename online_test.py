import pandas as pd
import numpy as np
import time
from Wiki_Trace_v2 import load_contiguous_chunk, OnlineLRBCache, summarize, EVAL_CAPACITY


chunk2 = load_contiguous_chunk("./cache-u-00.tsv", chunk_size=500_000, skip_rows=500_000)
chunk2 = chunk2.dropna().sort_values('timestamp').reset_index(drop=True)
chunk2['timestamp'] = chunk2['timestamp'].astype('int64')
chunk2['hash']      = chunk2['hash'].astype('int64')
chunk2['size']      = chunk2['size'].astype('int64')

print(f"Starting Online LRB Simulation on rows 500K - 1M...")

start_time = time.time()
online_cache = OnlineLRBCache(capacity_bytes=EVAL_CAPACITY, window_size=128_000, batch_size=128_000)

for _, row in chunk2.iterrows():
    online_cache.request(row['hash'], row['size'], row['firstbyte'])

duration = time.time() - start_time

hr = online_cache.hits / (online_cache.hits + online_cache.misses) if (online_cache.hits + online_cache.misses) > 0 else 0
bmr = online_cache.byte_misses / online_cache.total_bytes if online_cache.total_bytes > 0 else 0

print(f"\nCompleted in {duration:.1f} seconds.")
summarize("Online LRB (Rows 500K-1M)", hr, online_cache.latencies, online_cache.byte_misses, online_cache.total_bytes)
