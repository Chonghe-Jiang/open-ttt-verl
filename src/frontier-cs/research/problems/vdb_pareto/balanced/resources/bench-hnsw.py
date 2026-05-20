# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import os
import importlib
import faiss
import numpy as np

try:
    from faiss.contrib.datasets_fb import DatasetSIFT1M
except ImportError:
    from faiss.contrib.datasets import DatasetSIFT1M

# from datasets import load_sift1M


k = int(sys.argv[1])
todo = sys.argv[2:]

print("load data")

# xb, xq, xt, gt = load_sift1M()

ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()
xt = ds.get_train()

nq, d = xq.shape

# simple index cache utilities
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, 'data', 'index_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(name):
    return os.path.join(CACHE_DIR, name)

def _maybe_load_index(path):
    if os.path.exists(path):
        try:
            return faiss.read_index(path)
        except Exception:
            return None
    return None

def _save_index(index, path):
    try:
        faiss.write_index(index, path)
    except Exception:
        # ignore persistence errors to not block benchmarking
        pass

if todo == []:
    todo = 'hnsw hnsw_sq ivf ivf_hnsw_quantizer kmeans kmeans_hnsw nsg gpt5hnsw gpt5hnsw_paper gpt5hnsw_paper_fast'.split()


def evaluate(index):
    # batched search timing (using current threading configuration)
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    total_ms = (t1 - t0) * 1000.0
    per_query_ms = total_ms / float(nq)
    print('nq', nq)
    print("\t %7.3f ms total, %7.3f ms/query, R@1 %.4f, missing rate %.4f" % (
        total_ms, per_query_ms, recall_at_1, missing_rate))

    # single-request average latency with 1 thread over up to 100 queries
    prev_threads = None
    try:
        try:
            prev_threads = faiss.omp_get_max_threads()
        except Exception:
            prev_threads = None
        try:
            faiss.omp_set_num_threads(1)
        except Exception:
            pass
        num_samples = min(1000, nq)
        single_total = 0.0
        for i in range(num_samples):
            q = xq[i:i+1]
            s0 = time.time()
            index.search(q, k)
            s1 = time.time()
            single_total += (s1 - s0)
        single_total_ms = (single_total) * 1000.0
        print('num_samples', num_samples)
        print("\t single-request and total time over %d queries (1 thread over %d): %7.3f ms" % (num_samples, num_samples, single_total_ms))
    finally:
        if prev_threads is not None:
            try:
                faiss.omp_set_num_threads(prev_threads)
            except Exception:
                pass
    ## print failed if top recall at 1 < 0.2
    if float(recall_at_1) < 0.2:
        print("failed if top  recall at 1 < 0.2")

    return total_ms, recall_at_1, single_total_ms
    


if 'hnsw' in todo:

    print("Testing HNSW Flat")

    cache_path = _cache_path('hnsw_M32.faissindex')
    index = _maybe_load_index(cache_path)
    if index is None:
        index = faiss.IndexHNSWFlat(d, 32)
        # this is the default, higher is more accurate and slower to construct
        index.hnsw.efConstruction = 40
        print("add")
        index.verbose = True  # to see progress
        index.add(xb)
        _save_index(index, cache_path)
    else:
        print("loaded cached index:", cache_path)

    print("search")
    batch_pairs = []
    single_pairs = []
    for efSearch in 2,4,8,16, 32, 64, 128, 256, 512:
        for bounded_queue in [False]:
            print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
            index.hnsw.search_bounded_queue = bounded_queue
            index.hnsw.efSearch = efSearch
            total_ms, r1, single_total_ms = evaluate(index)
            batch_pairs.append((total_ms, r1))
            single_pairs.append((single_total_ms, r1))
    print("batch_time_vs_recall:", batch_pairs)
    print("single_time_vs_recall:", single_pairs)

if 'hnsw_sq' in todo:

    print("Testing HNSW with a scalar quantizer")
    # also set M so that the vectors and links both use 128 bytes per
    # entry (total 256 bytes)
    cache_path = _cache_path('hnsw_sq_M16_qt8.faissindex')
    index = _maybe_load_index(cache_path)
    if index is None:
        index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 16)
        print("training")
        index.train(xt)
        index.hnsw.efConstruction = 40
        print("add")
        index.verbose = True
        index.add(xb)
        _save_index(index, cache_path)
    else:
        print("loaded cached index:", cache_path)

    print("search")
    batch_pairs = []
    single_pairs = []
    for efSearch in 2,4,8,16, 32, 64, 128, 256, 512:
        print("efSearch", efSearch, end=' ')
        index.hnsw.efSearch = efSearch
        total_ms, r1, single_total_ms = evaluate(index)
        batch_pairs.append((total_ms, r1))
        single_pairs.append((single_total_ms, r1))
    print("batch_time_vs_recall:", batch_pairs)
    print("single_time_vs_recall:", single_pairs)

if 'ivf' in todo:

    print("Testing IVF Flat (baseline)")
    cache_path = _cache_path('ivf_flat_nlist16384.faissindex')
    index = _maybe_load_index(cache_path)
    if index is None:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 16384)
        index.cp.min_points_per_centroid = 5   # quiet warning
        index.verbose = True  # to see progress
        print("training")
        index.train(xt)
        print("add")
        index.add(xb)
        _save_index(index, cache_path)
    else:
        print("loaded cached index:", cache_path)

    print("search")
    batch_pairs = []
    single_pairs = []
    for nprobe in 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512:
        print("nprobe", nprobe, end=' ')
        index.nprobe = nprobe
        total_ms, r1, single_total_ms = evaluate(index)
        batch_pairs.append((total_ms, r1))
        single_pairs.append((single_total_ms, r1))
    print("batch_time_vs_recall:", batch_pairs)
    print("single_time_vs_recall:", single_pairs)

if 'ivf_hnsw_quantizer' in todo:

    print("Testing IVF Flat with HNSW quantizer")
    cache_path = _cache_path('ivf_flat_hnswq_nlist16384.faissindex')
    index = _maybe_load_index(cache_path)
    if index is None:
        quantizer = faiss.IndexHNSWFlat(d, 32)
        index = faiss.IndexIVFFlat(quantizer, d, 16384)
        index.cp.min_points_per_centroid = 5   # quiet warning
        index.quantizer_trains_alone = 2
        index.verbose = True  # to see progress
        print("training")
        index.train(xt)
        print("add")
        index.add(xb)
        _save_index(index, cache_path)
    else:
        print("loaded cached index:", cache_path)

    print("search")
    # set efSearch on the quantizer (works for both fresh and cached index)
    try:
        index.quantizer.hnsw.efSearch = 64
    except Exception:
        pass
    batch_pairs = []
    single_pairs = []
    for nprobe in 1, 2, 4, 8, 16, 32, 64, 128, 256, 512:
        print("nprobe", nprobe, end=' ')
        index.nprobe = nprobe
        total_ms, r1, single_total_ms = evaluate(index)
        batch_pairs.append((total_ms, r1))
        single_pairs.append((single_total_ms, r1))
    print("batch_time_vs_recall:", batch_pairs)
    print("single_time_vs_recall:", single_pairs)


if 'nsg' in todo:

    print("Testing NSG Flat")

    cache_path = _cache_path('nsg_M32.faissindex')
    index = _maybe_load_index(cache_path)
    if index is None:
        index = faiss.IndexNSGFlat(d, 32)
        index.build_type = 1
        print("add")
        index.verbose = True  # to see progress
        index.add(xb)
        _save_index(index, cache_path)
    else:
        print("loaded cached index:", cache_path)

    print("search")
    batch_pairs = []
    single_pairs = []
    for search_L in -1, 2, 4, 8, 16, 32, 64, 128, 256, 512:
        print("search_L", search_L, end=' ')
        index.nsg.search_L = search_L
        total_ms, r1, single_total_ms = evaluate(index)
        batch_pairs.append((total_ms, r1))
        single_pairs.append((single_total_ms, r1))
    print("batch_time_vs_recall:", batch_pairs)
    print("single_time_vs_recall:", single_pairs)

if 'gpt5hnsw' in todo:

    print("Testing GPT5HNSW (toy Python implementation)")
    from gpt5_hnsw import GPT5HNSW

    cache_path = _cache_path('gpt5hnsw_vectors.npy')
    index = None
    # load cached data if present (toy persistence for demo)
    index = GPT5HNSW(d, M=16, ef_search=64)
    print("add")
    index.add(xb)


    print("search")
    batch_pairs = []
    single_pairs = []
    for ef in 2, 4, 8, 16, 32, 64, 128:
        index.ef_search = ef
        print("efSearch", ef, end=' ')
        total_ms, r1, single_total_ms = evaluate(index)
        batch_pairs.append((total_ms, r1))
        single_pairs.append((single_total_ms, r1))
    print("batch_time_vs_recall:", batch_pairs)
    print("single_time_vs_recall:", single_pairs)

if 'gpt5hnsw_paper' in todo:

    print("Testing GPT5HNSW (paper-based hierarchical implementation)")
    from gpt5_hnsw_paper import GPT5HNSWPaper

    cache_path = _cache_path('gpt5hnsw_paper_vectors.npy')
    index = None
    if os.path.exists(cache_path):
        print("loaded cached vectors:", cache_path)
        data = np.load(cache_path)
        index = GPT5HNSWPaper(d, M=16, ef_construction=200, ef_search=64)
        index.add(data.astype('float32'))
    else:
        index = GPT5HNSWPaper(d, M=16, ef_construction=200, ef_search=64)
        print("add")
        index.add(xb)
        try:
            np.save(cache_path, xb)
        except Exception:
            pass

    print("search")
    batch_pairs = []
    single_pairs = []
    for ef in 8, 16, 32, 64, 128, 256:
        index.ef_search = ef
        print("efSearch", ef, end=' ')
        total_ms, r1, single_total_ms = evaluate(index)
        batch_pairs.append((total_ms, r1))
        single_pairs.append((single_total_ms, r1))
    print("batch_time_vs_recall:", batch_pairs)
    print("single_time_vs_recall:", single_pairs)

if 'gpt5hnsw_paper_fast' in todo:

    print("Testing GPT5HNSW (paper fast variant)")
    from gpt5_hnsw_paper_fast import GPT5HNSWPaperFast

    cache_path = _cache_path('gpt5hnsw_paper_fast_vectors.npy')
    index = None
    if os.path.exists(cache_path):
        print("loaded cached vectors:", cache_path)
        data = np.load(cache_path)
        index = GPT5HNSWPaperFast(d, M=16, ef_construction=96, ef_search=64)
        index.add(data.astype('float32'))
    else:
        index = GPT5HNSWPaperFast(d, M=16, ef_construction=96, ef_search=64)
        print("add")
        index.add(xb)
        try:
            np.save(cache_path, xb)
        except Exception:
            pass

    print("search")
    batch_pairs = []
    single_pairs = []
    for ef in 8, 16, 32, 64, 128, 256:
        index.ef_search = ef
        print("efSearch", ef, end=' ')
        total_ms, r1, single_total_ms = evaluate(index)
        batch_pairs.append((total_ms, r1))
        single_pairs.append((single_total_ms, r1))
    print("batch_time_vs_recall:", batch_pairs)
    print("single_time_vs_recall:", single_pairs)

# Minimal plugin: load a custom index class via CLI arg like
#   python bench-hnsw.py 1 custom:my_llm_index.LLMIndex
# The class should implement: __init__(dim, **kwargs), add(xb), search(xq, k)
custom_specs = [t for t in todo if isinstance(t, str) and t.startswith('custom:')]
for spec in custom_specs:
    try:
        print("Testing custom index:", spec)
        module_class = spec.split(':', 1)[1]
        module_name, class_name = module_class.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        Cls = getattr(mod, class_name)
        # Create the index; pass only dimension by default
        index = Cls(d)
        print("add")
        index.add(xb)

        print("search")
        batch_pairs = []
        single_pairs = []
        # If index exposes ef_search, sweep it; otherwise measure once
        if hasattr(index, 'ef_search'):
            grid = [8, 16, 32, 64, 128]
            for ef in grid:
                try:
                    index.ef_search = ef
                except Exception:
                    pass
                print("efSearch", ef, end=' ')
                total_ms, r1, single_total_ms = evaluate(index)
                batch_pairs.append((total_ms, r1))
                single_pairs.append((single_total_ms, r1))
        else:
            total_ms, r1, single_total_ms = evaluate(index)
            batch_pairs.append((total_ms, r1))
            single_pairs.append((single_total_ms, r1))

        print("batch_time_vs_recall:", batch_pairs)
        print("single_time_vs_recall:", single_pairs)
    except Exception as e:
        print("Custom index failed:", e)