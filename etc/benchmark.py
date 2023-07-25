#!/usr/bin/env python3

from argparse import ArgumentParser
from k_means_constrained import KMeansConstrained, __version__
import numpy as np
import time

p = ArgumentParser()
p.add_argument("-n", "--data-points", type=int, help="Number of data-points")
p.add_argument("-d", "--dimensions", type=int, help="Number of dimensions/features each data-point has")
p.add_argument("-K", "--clusters", type=int, help="Number of clusters")
p.add_argument("-ge", "--min-cluster-size", default=None, type=int, help="Minimum number of clusters assigned to each data-point")
p.add_argument("-le", "--max-cluster-size", default=None, type=int, help="Maximum number of clusters assigned to each data-point")
p.add_argument("-s", "--seed", type=int, default=42, help="Random state seed")
p.add_argument("-i", "--info", action='store_true', default=False , help="Print system info. `cpuinfo` is required to be installed.")
args = p.parse_args()

print("K-mean-constrained benchmark: data-points={args.n}, dimensions={args.d}, clusters={args.K}, min-cluster-size={args.ge}, max-cluster-size={args.le}, seed={args.s}")

if args.i:
    import scipy, ortools, joblib, platform, cpuinfo
    print(f"OS: {platform.platform()}")
    print(f"CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
    print(f"k-means-constrained version: {__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"scipy version: {scipy.__version__}")
    print(f"scipy version: {ortools.__version__}")
    print(f"joblib version: {joblib.__version__}")

np.random.seed(args.seed)

X = np.random.rand(args.n, args.d)

t = time.perf_counter()
clf = KMeansConstrained(
     n_clusters=args.K,
     size_min=args.ge,
     size_max=args.le,
     random_state=args.seed+1
 )
clf.fit_predict(X)

total_time = time.perf_counter() - t
print(f"Total time: {total_time:.2} seconds")

