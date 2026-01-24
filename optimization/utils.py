"""
utils.py

Authors: dkochar, natelgrw
Last Edited: 01/15/2026

Part of the TURBO Bayesian optimization framework.
Provides utility functions for space transformations and Latin hypercube sampling.
"""

###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import numpy as np


# ===== Space Transformation Utilities ===== #


def to_unit_cube(x, lb, ub):
    """
    Project points from bounded hypercube to [0, 1]^d.

    Normalizes input points from their original bounds to the unit hypercube,
    useful for standardizing inputs to optimization algorithms.

    Parameters:
    -----------
    x : numpy array
        Points to project, shape (n_points, d).
    lb : numpy array
        Lower bounds for each dimension, shape (d,).
    ub : numpy array
        Upper bounds for each dimension, shape (d,).
    
    Returns:
    --------
    numpy array
        Normalized points in [0, 1]^d, shape (n_points, d).
    
    Raises:
    -------
    AssertionError
        If bounds are invalid or input dimensions mismatch.
    """
    assert np.all(lb <= ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """
    Project points from [0, 1]^d to bounded hypercube.

    Denormalizes points from the unit hypercube back to their original
    bounds, reversing the to_unit_cube transformation.

    Parameters:
    -----------
    x : numpy array
        Points in [0, 1]^d, shape (n_points, d).
    lb : numpy array
        Lower bounds for each dimension, shape (d,).
    ub : numpy array
        Upper bounds for each dimension, shape (d,).
    
    Returns:
    --------
    numpy array
        Points in the original bounded space, shape (n_points, d).
    
    Raises:
    -------
    AssertionError
        If bounds are invalid or input dimensions mismatch.
    """
    assert np.all(lb <= ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


# ===== Design of Experiments ===== #


def latin_hypercube(n_pts, dim):
    """
    Generate Latin hypercube sample in [0, 1]^dim.

    Produces a space-filling design with one point per row and column
    in each dimension. Includes random perturbations within cells for
    better distribution properties.

    Parameters:
    -----------
    n_pts : int
        Number of points to generate.
    dim : int
        Dimensionality of the space.
    
    Returns:
    --------
    numpy array
        Latin hypercube sample, shape (n_pts, dim), values in [0, 1].
    """
    X = np.zeros((n_pts, dim))
    
    # divide each dimension into n_pts cells and take centers
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    
    for i in range(dim):
        X[:, i] = centers[np.random.permutation(n_pts)]

    # add perturbations within each cell to increase randomness
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X
