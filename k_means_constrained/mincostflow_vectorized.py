#!/usr/bin/env python

import numpy as np
from ortools.graph.pywrapgraph import SimpleMinCostFlow


class SimpleMinCostFlowNonVectorized(SimpleMinCostFlow):

    def AddArcWithCapacityAndUnitCostVectorized(self, tail, head, capacity, unit_cost):
        l = tail.shape[0]
        assert head.shape[0] == l
        assert capacity.shape[0] == l
        assert unit_cost.shape[0] == l

        for i in range(l):
            self.AddArcWithCapacityAndUnitCostVectorized(tail[i], head[i], capacity[i], unit_cost[i])

    def SetNodeSupplyVectorized(self, node, supply):
        l = node.shape[0]
        assert len(supply) == l

        for i in range(l):
            self.SetNodeSupplyVectorized(node[i], supply[i])

    def FlowVectorized(self, arc):
        l = arc.shape[0]
        flow = np.zeros(l, dtype=arc.dtype)

        for i in range(l):
            flow[i] = self.FlowVectorized(arc[i])

        return flow

try:
    # Cython paths must be fully qualified
    from k_means_constrained.mincostflow_vectorized_ import \
        SimpleMinCostFlow_AddArcWithCapacityAndUnitCostVectorized, \
        SimpleMinCostFlow_SetNodeSupplyVectorized, \
        SimpleMinCostFlow_FlowVectorized


    class SimpleMinCostFlowVectorized(SimpleMinCostFlow):

        def AddArcWithCapacityAndUnitCostVectorized(self, tail, head, capacity, unit_cost):
            return SimpleMinCostFlow_AddArcWithCapacityAndUnitCostVectorized(self, tail, head, capacity, unit_cost)

        def SetNodeSupplyVectorized(self, node, supply):
            return SimpleMinCostFlow_SetNodeSupplyVectorized(self, node, supply)

        def FlowVectorized(self, arc):
            return SimpleMinCostFlow_FlowVectorized(self, arc)

except ImportError:
    # Cython not compiled default to non vectorised version
    SimpleMinCostFlowVectorized = SimpleMinCostFlowNonVectorized
