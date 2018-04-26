#!/usr/bin/env python

import pytest

import numpy as np
from k_means_constrained.mincostflow_cython import SimpleMinCostFlowNumpy

def test_SimpleMinCostFlowNumpy():

    min_cost_flow = SimpleMinCostFlowNumpy()

    a = np.arange(100, dtype='int32')

    min_cost_flow.AddArcWithCapacityAndUnitCostNumpy(a, a, a, a)
