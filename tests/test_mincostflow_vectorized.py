#!/usr/bin/env python

import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from ortools.graph.pywrapgraph import SimpleMinCostFlow
from sklearn.metrics import euclidean_distances

from k_means_constrained.k_means_constrained_ import minimum_cost_flow_problem_graph
from k_means_constrained.mincostflow_vectorized import SimpleMinCostFlowVectorized

def test_SimpleMinCostFlowVectorized_equivalence():
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

    ## Original version
    min_cost_flow = SimpleMinCostFlow()
    N_edges = edges.shape[0]
    N_nodes = len(supplies)

    for i in range(0, N_edges):
        min_cost_flow.AddArcWithCapacityAndUnitCost(int(edges[i, 0]), int(edges[i, 1]),
                                                    int(capacities[i]), int(costs[i]))

    for i in range(0, N_nodes):
        min_cost_flow.SetNodeSupply(i, int(supplies[i]))

    if min_cost_flow.Solve() != min_cost_flow.OPTIMAL:
        raise Exception('There was an issue with the min cost flow input.')

    labels_M = np.array([min_cost_flow.Flow(i) for i in range(n_X*n_C)]).reshape(n_X, n_C)


    ## Vectorised version
    min_cost_flow_vec = SimpleMinCostFlowVectorized()

    min_cost_flow_vec.AddArcWithCapacityAndUnitCostVectorized(edges[:,0], edges[:,1], capacities, costs)
    min_cost_flow_vec.SetNodeSupplyVectorized(np.arange(N_nodes, dtype='int32'), supplies)

    if min_cost_flow_vec.Solve() != min_cost_flow_vec.OPTIMAL:
        raise Exception('There was an issue with the min cost flow input.')

    labels_M_vec = min_cost_flow_vec.FlowVectorized(np.arange(n_X * n_C, dtype='int32')).reshape(n_X, n_C)

    ## Should be equivalence
    assert_array_equal(labels_M, labels_M_vec)
