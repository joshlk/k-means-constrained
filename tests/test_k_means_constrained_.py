#!/usr/bin/env python

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_almost_equal
from sklearn import datasets

from sklearn.metrics import euclidean_distances
from sklearn.utils.extmath import row_norms
from sklearn.cluster.k_means_ import KMeans, _labels_inertia

from k_means_constrained.k_means_constrained_ import minimum_cost_flow_problem_graph, solve_min_cost_flow_graph, \
    KMeansConstrained, _labels_constrained


def test_minimum_cost_flow_problem_graph():
    # Setup graph
    X = np.array([
        [0, 0],
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0],
        [4, 4]
    ])
    C = np.array([
        [0, 0],
        [4, 4]
    ])
    size_min, size_max = 3, 10

    D = euclidean_distances(X, C, squared=True)

    edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(X, C, D, size_min, size_max)

    assert edges.shape[0] == len(costs)
    assert edges.shape[0] == len(capacities)
    assert len(np.unique(edges)) == len(supplies)
    assert costs.sum() > 0
    assert supplies.sum() == 0


def test_solve_min_cost_flow_graph():
    # Setup graph
    X = np.array([
        [0, 0],
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0],
        [4, 4]
    ])
    C = np.array([
        [0, 0],
        [4, 4]
    ])
    size_min, size_max = 3, 10

    D = euclidean_distances(X, C, squared=True)

    edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(X, C, D, size_min, size_max)
    labels = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

    cluster_size = pd.Series(labels).value_counts()

    assert (cluster_size > size_max).sum() == 0
    assert (cluster_size < size_min).sum() == 0


def test__labels_constrained():
    # Setup graph
    X = np.array([
        [0, 0],
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0],
        [4, 4]
    ])
    centers = np.array([
        [0, 0],
        [4, 4]
    ])
    size_min, size_max = 3, 10

    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    labels, inertia = _labels_constrained(X, centers, size_min, size_max, distances)

    # Labels
    cluster_size = pd.Series(labels).value_counts()
    assert (cluster_size > size_max).sum() == 0
    assert (cluster_size < size_min).sum() == 0

    # Distances
    assert distances.sum() > 0

    # Inertia
    assert inertia > 0


def test__labels_constrained_kmeans_parity():
    X = np.array([
        [0, 0],
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0],
        [4, 4]
    ]).astype('float')
    centers = np.array([
        [0, 0],
        [4, 4]
    ]).astype('float')
    size_min, size_max = 0, len(X)  # No restrictions and so should be the same as K-means

    x_squared_norms = row_norms(X, squared=True)

    distances_constrained = np.zeros(shape=(X.shape[0],), dtype=X.dtype)
    labels_constrained, inertia_constrained = _labels_constrained(X, centers, size_min, size_max, distances_constrained)

    distances_kmeans = np.zeros(shape=(X.shape[0],), dtype=X.dtype)
    labels_kmeans, inertia_kmeans = \
        _labels_inertia(X=X, x_squared_norms=x_squared_norms, centers=centers, precompute_distances=False,
                        distances=distances_kmeans, sample_weight=None)

    assert_array_equal(labels_constrained, labels_kmeans)
    assert_almost_equal(distances_constrained, distances_kmeans)
    assert inertia_constrained == inertia_kmeans


def test_KMeansConstrained():
    X = np.array([
        [0, 0],
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0],
        [3, 0],
        [4, 4]
    ])

    k = 3
    size_min, size_max = 3, 7

    clf = KMeansConstrained(
        n_clusters=k,
        size_min=size_min,
        size_max=size_max
    )

    clf.fit(X)
    y = clf.fit_predict(X)

    # Labels
    cluster_size = pd.Series(y).value_counts()
    assert (cluster_size > size_max).sum() == 0
    assert (cluster_size < size_min).sum() == 0


def test_KMeansConstrained_n_jobs():
    X, _ = datasets.make_blobs(n_samples=100, n_features=5, centers=10, random_state=1)

    n_jobs = -1
    k = 20
    size_min, size_max = 3, 40

    clf = KMeansConstrained(
        n_clusters=k,
        size_min=size_min,
        size_max=size_max,
        n_jobs=n_jobs
    )

    clf.fit(X)
    y = clf.fit_predict(X)

    # Labels
    cluster_size = pd.Series(y).value_counts()
    assert (cluster_size > size_max).sum() == 0
    assert (cluster_size < size_min).sum() == 0


def test_KMeansConstrained_parity_digits():

    iris = datasets.load_iris()
    X = iris.data

    k = 8
    random_state = 1
    size_min, size_max = None, None  # No restrictions and so should produce same result


    clf_constrained = KMeansConstrained(
        n_clusters=k,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state
    )
    y_constrained = clf_constrained.fit_predict(X)

    clf_kmeans = KMeans(
        n_clusters=k,
        random_state=random_state
    )
    y_kmeans = clf_kmeans.fit_predict(X)

    assert_array_equal(y_constrained, y_kmeans)
    assert_almost_equal(clf_constrained.cluster_centers_, clf_kmeans.cluster_centers_)
    assert_almost_equal(clf_constrained.inertia_, clf_kmeans.inertia_)


def test_KMeansConstrained_performance():

    n_cluster = 10
    n_X = 1000
    d = 3
    seed = 1

    np.random.seed(seed=seed)
    X = np.random.rand(n_X, d)
    clf = KMeansConstrained(n_cluster, size_min=None, size_max=None,
                            init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                            verbose=False, random_state=seed, copy_x=True, n_jobs=1)
    y = clf.fit_predict(X)
    #time = timeit('y = clf.fit_predict(X)', number=1, globals=globals())