"""Concurrent use of KMeansConstrained must be safe and give results
identical to serial execution."""
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from k_means_constrained import KMeansConstrained

N_THREADS = 4
N_TASKS = 8


def _make_data(seed, n=200, d=5):
    return np.random.RandomState(seed).rand(n, d)


def _fit(X, seed):
    clf = KMeansConstrained(n_clusters=4, size_min=20, size_max=80,
                            n_init=2, random_state=seed)
    labels = clf.fit_predict(X)
    return labels, clf.cluster_centers_, clf.inertia_


def test_concurrent_instances_distinct_data():
    datasets = [_make_data(seed) for seed in range(N_TASKS)]

    serial = [_fit(X, seed) for seed, X in enumerate(datasets)]

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        threaded = list(ex.map(lambda args: _fit(*args),
                               [(X, seed) for seed, X in enumerate(datasets)]))

    for (l1, c1, i1), (l2, c2, i2) in zip(serial, threaded):
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_allclose(c1, c2)
        assert i1 == pytest.approx(i2)


def test_concurrent_instances_shared_data():
    X = _make_data(0)
    X_before = X.copy()

    serial = [_fit(X, seed) for seed in range(N_TASKS)]

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        threaded = list(ex.map(lambda seed: _fit(X, seed), range(N_TASKS)))

    np.testing.assert_array_equal(X, X_before)

    for (l1, c1, i1), (l2, c2, i2) in zip(serial, threaded):
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_allclose(c1, c2)
        assert i1 == pytest.approx(i2)


def test_concurrent_predict_shared_estimator():
    X = _make_data(0)
    clf = KMeansConstrained(n_clusters=4, size_min=20, size_max=80,
                            n_init=2, random_state=0)
    clf.fit(X)
    expected = clf.predict(X)

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        results = list(ex.map(lambda _: clf.predict(X), range(N_TASKS)))

    for labels in results:
        np.testing.assert_array_equal(labels, expected)


@pytest.mark.skipif(not hasattr(sys, "_is_gil_enabled"),
                    reason="GIL status introspection requires Python >= 3.13")
def test_own_extensions_declare_freethreading_support():
    """No GIL re-enablement warning may name this package's extensions.
    A warning naming ortools is expected until fixed upstream."""
    import subprocess
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    import k_means_constrained\n"
        "gil_msgs = [str(x.message) for x in w if 'GIL' in str(x.message)]\n"
        "own = [m for m in gil_msgs if 'k_means_constrained' in m]\n"
        "assert not own, 'own extensions re-enabled the GIL: %r' % own\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
