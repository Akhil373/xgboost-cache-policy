import pandas as pd
import numpy as np
from collections import OrderedDict
import random

def compute_miss_penalty(size, firstbyte):
    return (firstbyte * 1000.0) + (size / 10_000_000.0 * 1000.0)

def simulate_ml(hashes, sizes, firstbytes, penalties, total_bytes_all,
                predictions, capacity, start_eval_idx=0, sample_size=64, use_bloom=False):
    seen       = set()
    cache_d    = {}  
    cache_l    = []   
    cur_bytes  = 0
    hits = misses = 0
    latencies  = []
    byte_misses = 0
    latest_pred = {}

    for i in range(len(hashes)):
        h, size, fb = hashes[i], sizes[i], firstbytes[i]
        latest_pred[h] = predictions[i]
        penalty        = penalties[i]
        
        is_val = i >= start_eval_idx

        if size > capacity:
            if is_val: 
                misses += 1; byte_misses += size; latencies.append(penalty)
            if use_bloom: seen.add(h)
            continue

        if h in cache_d:
            if is_val: hits += 1; latencies.append(0)
        else:
            if is_val: 
                misses += 1; byte_misses += size; latencies.append(penalty)

            if use_bloom and h not in seen:
                seen.add(h)
                continue
            if use_bloom:
                seen.add(h)

            while cur_bytes + size > capacity and cache_l:
                cands = random.sample(cache_l, min(sample_size, len(cache_l)))
                ev = max(cands, key=lambda k: latest_pred[k])
                
                idx = cache_d[ev][2]
                last = cache_l[-1]
                cache_l[idx] = last
                cache_d[last] = (cache_d[last][0], cache_d[last][1], idx)
                cache_l.pop()
                cur_bytes -= cache_d[ev][0]
                del cache_d[ev]

            cache_l.append(h)
            cache_d[h] = (size, fb, len(cache_l) - 1)
            cur_bytes += size

    hr = hits / (hits + misses) if hits + misses else 0
    return hr, latencies, byte_misses, total_bytes_all
