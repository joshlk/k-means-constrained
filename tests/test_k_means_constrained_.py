#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix, issparse

from k_means_constrained.sklearn_import.metrics.pairwise import euclidean_distances

from k_means_constrained.k_means_constrained_ import minimum_cost_flow_problem_graph, solve_min_cost_flow_graph, \
    KMeansConstrained, _labels_constrained

def sort_coordinates(array):
    array = array[np.lexsort(np.fliplr(array).T)]
    return array


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

    y = clf.fit_predict(X)

    # Labels
    cluster_size = pd.Series(y).value_counts()
    assert (cluster_size > size_max).sum() == 0
    assert (cluster_size < size_min).sum() == 0


def test_KMeansConstrained_predict_method():
    X = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
    ])

    k = 2
    size_max = 2

    clf = KMeansConstrained(
        n_clusters=k,
        size_max=size_max
    )

    clf.fit(X)

    y_constrained = clf.predict(X)  # Expected np.array([0, 0, 1, 1])
    y_normal = super(KMeansConstrained, clf).predict(X)  # Expected np.array([0, 0, 0, 1])

    cluster_size_constrained = pd.Series(y_constrained).value_counts()
    assert (cluster_size_constrained > size_max).any() == False
    assert len(cluster_size_constrained) == k

    cluster_size_normal = pd.Series(y_normal).value_counts()
    assert (cluster_size_normal > size_max).any() == True
    assert len(cluster_size_normal) == k


def test_spare_not_implemented():
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

    X = csc_matrix(X)

    with pytest.raises(NotImplementedError):
        clf.fit(X)

    with pytest.raises(NotImplementedError):
        clf.fit_predict(X)

#######
# Parity tests only works with sklearn v0.19.2 but does not run on Python 3.8+
#######

# from sklearn.cluster import KMeans
# from sklearn.cluster.k_means_ import _labels_inertia
# from numpy.testing import assert_array_equal, assert_almost_equal
# from k_means_constrained.sklearn_import.utils.extmath import row_norms

# Test passes on Python 3.7
# def test__labels_constrained_kmeans_parity():
#     X = np.array([
#         [0, 0],
#         [1, 2],
#         [1, 4],
#         [1, 0],
#         [4, 2],
#         [4, 4],
#         [4, 0],
#         [4, 4]
#     ]).astype('float')
#     centers = np.array([
#         [0, 0],
#         [4, 4]
#     ]).astype('float')
#     size_min, size_max = 0, len(X)  # No restrictions and so should be the same as K-means
#
#     x_squared_norms = row_norms(X, squared=True)
#
#     distances_constrained = np.zeros(shape=(X.shape[0],), dtype=X.dtype)
#     labels_constrained, inertia_constrained = _labels_constrained(X, centers, size_min, size_max, distances_constrained)
#
#     distances_kmeans = np.zeros(shape=(X.shape[0],), dtype=X.dtype)
#     labels_kmeans, inertia_kmeans = \
#         _labels_inertia(X=X, x_squared_norms=x_squared_norms, centers=centers, precompute_distances=False,
#                         distances=distances_kmeans)
#
#     assert_array_equal(labels_constrained, labels_kmeans)
#     assert_almost_equal(distances_constrained, distances_kmeans)
#     assert inertia_constrained == inertia_kmeans

# Test passes on Python 3.7
# def test_KMeansConstrained_parity_digits():
#     iris = datasets.load_iris()
#     X = iris.data
#
#     k = 8
#     random_state = 1
#     size_min, size_max = None, None  # No restrictions and so should produce same result
#
#     clf_constrained = KMeansConstrained(
#         size_min=size_min,
#         size_max=size_max,
#         n_clusters=k,
#         random_state=random_state,
#         init='k-means++',
#         n_init=10,
#         max_iter=300,
#         tol=1e-4
#     )
#     y_constrained = clf_constrained.fit_predict(X)
#
#     # TODO: Testing scikit-learn has be set to v0.19. This is because there is a discrepancy scikit-learn v0.22 https://github.com/scikit-learn/scikit-learn/issues/16623
#     clf_kmeans = KMeans(
#         n_clusters=k,
#         random_state=random_state,
#         init='k-means++',
#         n_init=10,
#         max_iter=300,
#         tol=1e-4
#     )
#     y_kmeans = clf_kmeans.fit_predict(X)
#
#     # Each cluster should have the same number of datapoints assigned to it
#     constrained_ndp = pd.Series(y_constrained).value_counts().values
#     kmeans_ndp = pd.Series(y_kmeans).value_counts().values
#
#     assert_almost_equal(constrained_ndp, kmeans_ndp)
#
#     # Sort the cluster coordinates (otherwise in a random order)
#     constrained_cluster_centers = sort_coordinates(clf_constrained.cluster_centers_)
#     kmean_cluster_centers = sort_coordinates(clf_kmeans.cluster_centers_)
#
#     assert_almost_equal(constrained_cluster_centers, kmean_cluster_centers)


####
# Further tests removed as removed sklearn dependency
####

# from sklearn import datasets
#
# def test_KMeansConstrained_n_jobs():
#     X, _ = datasets.make_blobs(n_samples=100, n_features=5, centers=10, random_state=1)
#
#     n_jobs = -1
#     k = 20
#     size_min, size_max = 3, 40
#
#     clf = KMeansConstrained(
#         n_clusters=k,
#         size_min=size_min,
#         size_max=size_max,
#         n_jobs=n_jobs
#     )
#
#     y = clf.fit_predict(X)
#
#     # Labels
#     cluster_size = pd.Series(y).value_counts()
#     assert (cluster_size > size_max).sum() == 0
#     assert (cluster_size < size_min).sum() == 0