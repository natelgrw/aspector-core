"""
turbo_m.py

Authors: dkochar, natelgrw
Last Edited: 01/15/2026

TURBO-m multi-region Bayesian optimization algorithm.
Extends TURBO-1 to handle multiple parallel trust regions for batch 
optimization with adaptive restart capabilities.
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

import math
import sys
from copy import deepcopy

import numpy as np

from .gp import train_gp
from .turbo_1 import Turbo1
from .utils import from_unit_cube, latin_hypercube, to_unit_cube


# ===== TURBO-M Multi-Region Trust Region Optimizer ===== #


class TurboM(Turbo1):
    """
    Multi-trust-region Bayesian optimization using TURBO-m algorithm.

    Extends Turbo1 with multiple parallel trust regions for improved parallel
    evaluation and adaptive restart when regions converge. Each trust region
    maintains independent length scales and hyperparameters.

    Initialization Parameters:
    --------------------------
    f : callable
        Objective function to minimize. Takes numpy array of shape (d,) and returns float.
    lb : numpy array
        Lower bounds for each dimension, shape (d,).
    ub : numpy array
        Upper bounds for each dimension, shape (d,).
    n_init : int
        Number of initial Latin hypercube points per trust region (2*dim recommended).
    max_evals : int
        Total evaluation budget limit across all trust regions.
    n_trust_regions : int
        Number of parallel trust regions to maintain.
    batch_size : int, optional
        Number of points evaluated per batch (default: 1).
    verbose : bool, optional
        Print optimization progress information (default: True).
    use_ard : bool, optional
        Use Automatic Relevance Determination for GP kernel (default: True).
    max_cholesky_size : int, optional
        Maximum training points for Cholesky decomposition, else uses Lanczos (default: 2000).
    n_training_steps : int, optional
        Number of ADAM steps for hyperparameter optimization (default: 50).
    min_cuda : int, optional
        Minimum training points to use CUDA (default: 1024).
    device : str, optional
        Compute device: "cpu" or "cuda" (default: "cpu").
    dtype : str, optional
        Floating point precision: "float32" or "float64" (default: "float64").
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        n_trust_regions,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    ):

        self.n_trust_regions = n_trust_regions
        super().__init__(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=max_evals,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            max_cholesky_size=max_cholesky_size,
            n_training_steps=n_training_steps,
            min_cuda=min_cuda,
            device=device,
            dtype=dtype,
        )

        self.succtol = 3
        self.failtol = max(5, self.dim)

        # basic input checks
        assert n_trust_regions > 1 and isinstance(max_evals, int)
        assert max_evals > n_trust_regions * n_init, "Not enough trust regions to do initial evaluations"
        assert max_evals > batch_size, "Not enough evaluations to do a single batch"

        # remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        self._restart()


    # ==== Trust Region Initialization and Restart ==== #


    def _restart(self):
        """
        Reinitialize multi-region state for new optimization phase.

        Resets all region-specific state variables including:
        - Region assignment indices for all proposed points
        - Success and failure counters per region
        - Trust region length parameters for each region

        Called at initialization. Note: Individual regions can be restarted
        during optimization while others continue.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._idx = np.zeros((0, 1), dtype=int)  # Track what trust region proposed what using an index vector
        self.failcount = np.zeros(self.n_trust_regions, dtype=int)
        self.succcount = np.zeros(self.n_trust_regions, dtype=int)
        self.length = self.length_init * np.ones(self.n_trust_regions)

    def _adjust_length(self, fX_next, i):
        """
        Expand or contract trust region length for region i based on progress.

        Implements per-region adaptive trust region sizing:
        - Expansion: If succtol (3) consecutive improvements found, double region length
        - Contraction: If failtol (max(5, dim)) failures accumulated, halve region length
        - Improvement criterion: min(fX_next) < target - 1e-3 * |target|
        where target is minimum value previously found in region i

        Parameters
        ----------
        fX_next : np.ndarray
            Function evaluation results from candidates in region i. Shape (n_batch, 1).
        i : int
            Index of trust region to adjust (0 to n_trust_regions-1).

        Returns
        -------
        None
        """
        assert i >= 0 and i <= self.n_trust_regions - 1

        fX_min = self.fX[self._idx[:, 0] == i, 0].min()
        if fX_next.min() < fX_min - 1e-3 * math.fabs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(fX_next)

        if self.succcount[i] == self.succtol:
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:
            self.length[i] /= 2.0
            self.failcount[i] = 0


    # ==== Multi-Region Candidate Selection ==== #


    def _select_candidates(self, X_cand, y_cand):
        """
        Select best candidates from all trust regions simultaneously.

        Implements global batch point selection across all regions:
        For each batch slot, finds minimum predicted value across all regions and candidates,
        then masks that point to prevent duplicate selection in subsequent batch slots.
        This ensures diversity across regions in parallel evaluation.

        Parameters
        ----------
        X_cand : np.ndarray
            Candidate pools from all regions. Shape (n_trust_regions, n_cand, dim).
            Each region has independent candidate set.
        y_cand : np.ndarray
            Predicted values at candidates for each batch slot, all regions.
            Shape (n_trust_regions, n_cand, batch_size).

        Returns
        -------
        X_next : np.ndarray
            Selected points for evaluation. Shape (batch_size, dim).
        idx_next : np.ndarray
            Region indices indicating which region proposed each selected point.
            Shape (batch_size, 1).
        """
        assert X_cand.shape == (self.n_trust_regions, self.n_cand, self.dim)
        assert y_cand.shape == (self.n_trust_regions, self.n_cand, self.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.batch_size, self.dim))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        for k in range(self.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (self.n_trust_regions, self.n_cand))
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            idx_next[k, 0] = i
            assert np.isfinite(y_cand[i, j, k]) 
            y_cand[i, j, :] = np.inf

        return X_next, idx_next


    # ===== Main Multi-Region Optimization ===== #


    def optimize(self):
        """
        Run the complete multi-region trust region Bayesian optimization process.

        Implements the TURBO-M optimization loop with multiple parallel trust regions:
        1. Initializes each region with Latin hypercube design points
        2. Iteratively generates candidates from all regions using independent GPs
        3. Selects best candidates globally across all regions
        4. Updates region-specific trust region sizes based on progress
        5. Restarts converged regions (length < length_min) with fresh designs
        6. Tracks region assignments and maintains per-region hyperparameters

        Multi-region advantages:
        - Parallel exploration of different regions of search space
        - Graceful handling of converged regions through restarts
        - Better escape from local minima via multiple independent regions

        Global optimization history (self.X, self.fX, self._idx) accumulates all
        evaluations across all regions. Region-specific data accessed via self._idx.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # create initial points for each TR
        for i in range(self.n_trust_regions):
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = np.array([[self.f(x)] for x in X_init])

            # update budget and set as initial data for this TR
            self.X = np.vstack((self.X, X_init))
            self.fX = np.vstack((self.fX, fX_init))
            self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
            self.n_evals += self.n_init

            if self.verbose:
                fbest = fX_init.min()
                print(f"TR-{i} starting from: {fbest:.4}")
                sys.stdout.flush()

        # Thompson sample to get next suggestions
        while self.n_evals < self.max_evals:

            # generate candidates from each TR
            X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.dim))
            y_cand = np.inf * np.ones((self.n_trust_regions, self.n_cand, self.batch_size))
            for i in range(self.n_trust_regions):
                idx = np.where(self._idx == i)[0]  # Extract all "active" indices

                X = deepcopy(self.X[idx, :])
                X = to_unit_cube(X, self.lb, self.ub)

                fX = deepcopy(self.fX[idx, 0].ravel())

                n_training_steps = 0 if self.hypers[i] else self.n_training_steps

                # create new candidates
                X_cand[i, :, :], y_cand[i, :, :], self.hypers[i] = self._create_candidates(
                    X, fX, length=self.length[i], n_training_steps=n_training_steps, hypers=self.hypers[i]
                )

            # select the next candidates
            X_next, idx_next = self._select_candidates(X_cand, y_cand)
            assert X_next.min() >= 0.0 and X_next.max() <= 1.0

            X_next = from_unit_cube(X_next, self.lb, self.ub)

            # evaluate batch
            fX_next = np.array([[self.f(x)] for x in X_next])

            # update trust regions
            for i in range(self.n_trust_regions):
                idx_i = np.where(idx_next == i)[0]
                if len(idx_i) > 0:
                    self.hypers[i] = {}
                    fX_i = fX_next[idx_i]

                    if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * math.fabs(self.fX.min()):
                        n_evals, fbest = self.n_evals, fX_i.min()
                        print(f"{n_evals}) New best @ TR-{i}: {fbest:.4}")
                        sys.stdout.flush()
                    self._adjust_length(fX_i, i)

            # update budget and append data
            self.n_evals += self.batch_size
            self.X = np.vstack((self.X, deepcopy(X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))
            self._idx = np.vstack((self._idx, deepcopy(idx_next)))

            # check if any TR needs to be restarted
            for i in range(self.n_trust_regions):
                if self.length[i] < self.length_min:
                    idx_i = self._idx[:, 0] == i

                    if self.verbose:
                        n_evals, fbest = self.n_evals, self.fX[idx_i, 0].min()
                        print(f"{n_evals}) TR-{i} converged to: : {fbest:.4}")
                        sys.stdout.flush()

                    # reset length and counters, remove old data from trust region
                    self.length[i] = self.length_init
                    self.succcount[i] = 0
                    self.failcount[i] = 0
                    self._idx[idx_i, 0] = -1
                    self.hypers[i] = {}

                    # create a new initial design
                    X_init = latin_hypercube(self.n_init, self.dim)
                    X_init = from_unit_cube(X_init, self.lb, self.ub)
                    fX_init = np.array([[self.f(x)] for x in X_init])

                    # print progress
                    if self.verbose:
                        n_evals, fbest = self.n_evals, fX_init.min()
                        print(f"{n_evals}) TR-{i} is restarting from: : {fbest:.4}")
                        sys.stdout.flush()

                    # append data to local history
                    self.X = np.vstack((self.X, X_init))
                    self.fX = np.vstack((self.fX, fX_init))
                    self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
                    self.n_evals += self.n_init
