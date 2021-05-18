# Tests copied and modified from: https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/cluster/tests/test_k_means.py

import sys
import numpy as np
from numpy.testing import assert_equal, assert_warns, assert_array_almost_equal, assert_array_equal, assert_raises,\
    assert_raises_regex
from k_means_constrained.k_means_constrained_ import k_means_constrained, _labels_constrained
from k_means_constrained import KMeansConstrained
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import v_measure_score
from unittest import SkipTest

from sklearn.utils._testing import assert_raise_message

# non centered, sparse centers to check the
centers = np.array([
    [0.0, 5.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 4.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 5.0, 1.0],
])
n_samples = 100
n_clusters, n_features = centers.shape
X, true_labels = make_blobs(n_samples=n_samples, centers=centers,
                            cluster_std=1., random_state=42)

def test_labels_assignment_and_inertia():
    # pure numpy implementation as easily auditable reference gold
    # implementation
    rng = np.random.RandomState(42)
    noisy_centers = centers + rng.normal(size=centers.shape)
    labels_gold = - np.ones(n_samples, dtype=int)
    mindist = np.empty(n_samples)
    mindist.fill(np.infty)
    for center_id in range(n_clusters):
        dist = np.sum((X - noisy_centers[center_id]) ** 2, axis=1)
        labels_gold[dist < mindist] = center_id
        mindist = np.minimum(dist, mindist)
    inertia_gold = mindist.sum()
    assert (mindist >= 0.0).all()
    assert (labels_gold != -1).all()

    # perform label assignment using the dense array input
    distances = np.zeros(shape=(len(X),), dtype=X.dtype)
    labels_array, inertia_array = _labels_constrained(
        X, noisy_centers, size_min=0, size_max=len(X), distances=distances)
    assert_array_almost_equal(inertia_array, inertia_gold)
    assert_array_equal(labels_array, labels_gold)

def _check_fitted_model(km):
    # check that the number of clusters centers and distinct labels match
    # the expectation
    centers = km.cluster_centers_
    assert_equal(centers.shape, (n_clusters, n_features))

    labels = km.labels_
    assert_equal(np.unique(labels).shape[0], n_clusters)

    # check that the labels assignment are perfect (up to a permutation)
    assert_equal(v_measure_score(true_labels, labels), 1.0)
    assert km.inertia_ > 0.0

    # check error on dataset being too small
    assert_raises(ValueError, km.fit, [[0., 1.]])


def test_k_means_plus_plus_init():
    km = KMeansConstrained(init="k-means++", n_clusters=n_clusters,
                random_state=42).fit(X)
    _check_fitted_model(km)


def test_k_means_new_centers():
    # Explore the part of the code where a new center is reassigned
    X = np.array([[0, 0, 1, 1],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0]])
    labels = [0, 1, 2, 1, 1, 2]
    bad_centers = np.array([[+0, 1, 0, 0],
                            [.2, 0, .2, .2],
                            [+0, 0, 0, 0]])

    km = KMeansConstrained(n_clusters=3, init=bad_centers, n_init=1, max_iter=10,
                random_state=1)

    for i in range(2):
        km.fit(X)
        this_labels = km.labels_
        # Reorder the labels so that the first instance is in cluster 0,
        # the second in cluster 1, ...
        this_labels = np.unique(this_labels, return_index=True)[1][this_labels]
        np.testing.assert_array_equal(this_labels, labels)


def test_k_means_plus_plus_init_2_jobs():
    if sys.version_info[:2] < (3, 4):
        raise SkipTest(
            "Possible multi-process bug with some BLAS under Python < 3.4")

    km = KMeansConstrained(init="k-means++", n_clusters=n_clusters, n_jobs=2,
                random_state=42).fit(X)
    _check_fitted_model(km)


def test_k_means_random_init():
    km = KMeansConstrained(init="random", n_clusters=n_clusters, random_state=42)
    km.fit(X)
    _check_fitted_model(km)


def test_k_means_perfect_init():
    km = KMeansConstrained(init=centers.copy(), n_clusters=n_clusters, random_state=42,
                n_init=1)
    km.fit(X)
    _check_fitted_model(km)


def test_k_means_n_init():
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(40, 2))

    # two regression tests on bad n_init argument
    # previous bug: n_init <= 0 threw non-informative TypeError (#3858)
    assert_raises_regex(ValueError, "n_init", KMeansConstrained(n_init=0).fit, X)
    assert_raises_regex(ValueError, "n_init", KMeansConstrained(n_init=-1).fit, X)


def test_k_means_explicit_init_shape():
    # test for sensible errors when giving explicit init
    # with wrong number of features or clusters
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(40, 3))

    # mismatch of number of features
    km = KMeansConstrained(n_init=1, init=X[:, :2], n_clusters=len(X))
    msg = "does not match the number of features of the data"
    assert_raises_regex(ValueError, msg, km.fit, X)
    # for callable init
    km = KMeansConstrained(n_init=1,
               init=lambda X_, k, random_state: X_[:, :2],
               n_clusters=len(X))
    assert_raises_regex(ValueError, msg, km.fit, X)
    # mismatch of number of clusters
    msg = "does not match the number of clusters"
    km = KMeansConstrained(n_init=1, init=X[:2, :], n_clusters=3)
    assert_raises_regex(ValueError, msg, km.fit, X)
    # for callable init
    km = KMeansConstrained(n_init=1,
               init=lambda X_, k, random_state: X_[:2, :],
               n_clusters=3)
    assert_raises_regex(ValueError, msg, km.fit, X)


def test_k_means_fortran_aligned_data():
    # Check the KMeans will work well, even if X is a fortran-aligned data.
    X = np.asfortranarray([[0, 0], [0, 1], [0, 1]])
    centers = np.array([[0, 0], [0, 1]])
    labels = np.array([0, 1, 1])
    km = KMeansConstrained(n_init=1, init=centers,
                random_state=42, n_clusters=2)
    km.fit(X)
    assert_array_equal(km.cluster_centers_, centers)
    assert_array_equal(km.labels_, labels)


def test_k_means_invalid_init():
    km = KMeansConstrained(init="invalid", n_init=1, n_clusters=n_clusters)
    assert_raises(ValueError, km.fit, X)


def test_k_means_copyx():
    # Check if copy_x=False returns nearly equal X after de-centering.
    my_X = X.copy()
    km = KMeansConstrained(copy_x=False, n_clusters=n_clusters, random_state=42)
    km.fit(my_X)
    _check_fitted_model(km)

    # check if my_X is centered
    assert_array_almost_equal(my_X, X)


def test_k_means_non_collapsed():
    # Check k_means with a bad initialization does not yield a singleton
    # Starting with bad centers that are quickly ignored should not
    # result in a repositioning of the centers to the center of mass that
    # would lead to collapsed centers which in turns make the clustering
    # dependent of the numerical unstabilities.
    my_X = np.array([[1.1, 1.1], [0.9, 1.1], [1.1, 0.9], [0.9, 1.1]])
    array_init = np.array([[1.0, 1.0], [5.0, 5.0], [-5.0, -5.0]])
    km = KMeansConstrained(init=array_init, n_clusters=3, random_state=42, n_init=1)
    km.fit(my_X)

    # centers must not been collapsed
    assert_equal(len(np.unique(km.labels_)), 3)

    centers = km.cluster_centers_
    assert (np.linalg.norm(centers[0] - centers[1]) >= 0.1).all()
    assert (np.linalg.norm(centers[0] - centers[2]) >= 0.1).all()
    assert (np.linalg.norm(centers[1] - centers[2]) >= 0.1).all()


def test_predict():
    km = KMeansConstrained(n_clusters=n_clusters, random_state=42)

    km.fit(X)

    # sanity check: predict centroid labels
    pred = km.predict(km.cluster_centers_)
    assert_array_equal(pred, np.arange(n_clusters))

    # sanity check: re-predict labeling for training set samples
    pred = km.predict(X)
    assert_array_equal(pred, km.labels_)

    # re-predict labels for training set using fit_predict
    pred = km.fit_predict(X)
    assert_array_equal(pred, km.labels_)


def test_score():

    km1 = KMeansConstrained(n_clusters=n_clusters, max_iter=1, random_state=42, n_init=1)
    s1 = km1.fit(X).score(X)
    km2 = KMeansConstrained(n_clusters=n_clusters, max_iter=10, random_state=42, n_init=1)
    s2 = km2.fit(X).score(X)
    assert s2 > s1


def test_transform():
    km = KMeansConstrained(n_clusters=n_clusters)
    km.fit(X)
    X_new = km.transform(km.cluster_centers_)

    for c in range(n_clusters):
        assert_equal(X_new[c, c], 0)
        for c2 in range(n_clusters):
            if c != c2:
                assert X_new[c, c2] > 0


def test_fit_transform():
    X1 = KMeansConstrained(n_clusters=3, random_state=51).fit(X).transform(X)
    X2 = KMeansConstrained(n_clusters=3, random_state=51).fit_transform(X)
    assert_array_equal(X1, X2)


def test_n_init():
    # Check that increasing the number of init increases the quality
    n_runs = 5
    n_init_range = [1, 5, 10]
    inertia = np.zeros((len(n_init_range), n_runs))
    for i, n_init in enumerate(n_init_range):
        for j in range(n_runs):
            km = KMeansConstrained(n_clusters=n_clusters, init="random", n_init=n_init,
                        random_state=j).fit(X)
            inertia[i, j] = km.inertia_

    inertia = inertia.mean(axis=1)
    failure_msg = ("Inertia %r should be decreasing"
                   " when n_init is increasing.") % list(inertia)
    for i in range(len(n_init_range) - 1):
        assert (inertia[i] >= inertia[i + 1]).all(), failure_msg


def test_k_means_function():
    # test calling the k_means function directly
    # catch output
    old_stdout = sys.stdout
    #sys.stdout = StringIO()
    try:
        cluster_centers, labels, inertia = k_means_constrained(X, n_clusters=n_clusters,
                                                   verbose=True)
    finally:
        sys.stdout = old_stdout
    centers = cluster_centers
    assert_equal(centers.shape, (n_clusters, n_features))

    labels = labels
    assert_equal(np.unique(labels).shape[0], n_clusters)

    # check that the labels assignment are perfect (up to a permutation)
    assert_equal(v_measure_score(true_labels, labels), 1.0)
    assert inertia > 0.0

    # check warning when centers are passed
    assert_warns(RuntimeWarning, k_means_constrained, X, n_clusters=n_clusters,
                 init=centers)

    # to many clusters desired
    assert_raises(ValueError, k_means_constrained, X, n_clusters=X.shape[0] + 1)



def test_max_iter_error():

    km = KMeansConstrained(max_iter=-1)
    assert_raise_message(ValueError, 'Number of iterations should be',
                         km.fit, X)


def test_float_precision():
    km = KMeansConstrained(n_init=1, random_state=30)

    inertia = {}
    X_new = {}
    centers = {}

    for dtype in [np.float64, np.float32]:
        X_test = X.astype(dtype)
        km.fit(X_test)
        # dtype of cluster centers has to be the dtype of the input
        # data
        assert_equal(km.cluster_centers_.dtype, dtype)
        inertia[dtype] = km.inertia_
        X_new[dtype] = km.transform(X_test)
        centers[dtype] = km.cluster_centers_
        # ensure the extracted row is a 2d array
        assert_equal(km.predict(X_test[:1]),
                     km.labels_[0])
        if hasattr(km, 'partial_fit'):
            km.partial_fit(X_test[0:3])
            # dtype of cluster centers has to stay the same after
            # partial_fit
            assert_equal(km.cluster_centers_.dtype, dtype)

    # compare arrays with low precision since the difference between
    # 32 and 64 bit sometimes makes a difference up to the 4th decimal
    # place
    assert_array_almost_equal(inertia[np.float32], inertia[np.float64],
                              decimal=4)
    assert_array_almost_equal(X_new[np.float32], X_new[np.float64],
                              decimal=4)
    assert_array_almost_equal(centers[np.float32], centers[np.float64],
                              decimal=4)


def test_k_means_init_centers():
    # This test is used to check KMeans won't mutate the user provided input
    # array silently even if input data and init centers have the same type
    X_small = np.array([[1.1, 1.1], [-7.5, -7.5], [-1.1, -1.1], [7.5, 7.5]])
    init_centers = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, -5.0]])
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        X_test = dtype(X_small)
        init_centers_test = dtype(init_centers)
        assert_array_equal(init_centers, init_centers_test)
        km = KMeansConstrained(init=init_centers_test, n_clusters=3, n_init=1)
        km.fit(X_test)
        assert_equal(False, np.may_share_memory(km.cluster_centers_, init_centers))


def test_sparse_k_means_init_centers():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data

    # Get a local optimum
    centers = KMeansConstrained(n_clusters=3, size_min=50).fit(X).cluster_centers_

    # Fit starting from a local optimum shouldn't change the solution
    np.testing.assert_allclose(
        centers,
        KMeansConstrained(n_clusters=3, size_min=50,
               init=centers,
               n_init=1).fit(X).cluster_centers_
    )


def test_sparse_validate_centers():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data

    # Get a local optimum
    centers = KMeansConstrained(n_clusters=4).fit(X).cluster_centers_

    # Test that a ValueError is raised for validate_center_shape
    classifier = KMeansConstrained(n_clusters=3, init=centers, n_init=1)

    assert_raises(ValueError, classifier.fit, X)
