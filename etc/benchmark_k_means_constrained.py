#!/usr/bin/env python3

from argparse import ArgumentParser
import k_means_constrained
import numpy as np
import time
import logging
import os

p = ArgumentParser()
p.add_argument("-n", "--data-points", required=True, type=int, help="Number of data-points")
p.add_argument("-d", "--dimensions", required=True, type=int, help="Number of dimensions/features each data-point has")
p.add_argument("-K", "--clusters", required=True, type=int, help="Number of clusters")
p.add_argument("-ge", "--min-cluster-size", default=None, help="Minimum number of clusters assigned to each data-point")
p.add_argument("-le", "--max-cluster-size", default=None, help="Maximum number of clusters assigned to each data-point")
p.add_argument("-s", "--seed", type=int, default=42, help="Random state seed")
p.add_argument("-i", "--info", action='store_true', default=False , help="Print system info. `cpuinfo` is required to be installed.")
args = p.parse_args()


logging.basicConfig(
    level=os.environ.get('LOGLEVEL', 'DEBUG').upper()
)

print(f"K-mean-constrained benchmark: data-points={args.data_points}, dimensions={args.dimensions}, clusters={args.clusters}, min-cluster-size={args.min_cluster_size}, max-cluster-size={args.max_cluster_size}, seed={args.seed}")

if args.info:
    import scipy, ortools, joblib, platform, cpuinfo, sklearn, k_means_constrained
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    print(f"CPU cores: {cpuinfo.get_cpu_info()['count']}")
    print(f"k-means-constrained version: {k_means_constrained.__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"scipy version: {scipy.__version__}")
    print(f"ortools version: {ortools.__version__}")
    print(f"joblib version: {joblib.__version__}")
    print(f"sklearn version: {sklearn.__version__}")

np.random.seed(args.seed)

X = np.random.rand(args.data_points, args.dimensions)

t = time.perf_counter()
clf = k_means_constrained.KMeansConstrained(
     n_clusters=args.clusters,
     size_min=int(args.min_cluster_size) if args.min_cluster_size else None,
     size_max=int(args.max_cluster_size) if args.max_cluster_size else None,
     random_state=args.seed+1,
     #algorithm='lloyd', # implied
     init='k-means++',
     n_init=10,
     max_iter=300,
     tol=0.0001,
     n_jobs=10,
 )
clf.fit_predict(X)

total_time = time.perf_counter() - t
print(f"Total time: {total_time:.2f} seconds")

