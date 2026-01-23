"""
turbo_1.py

Authors: dkochar, natelgrw
Last Edited: 01/15/2026

TURBO-1 single-trust-region Bayesian optimization algorithm.
Implements efficient sequential optimization using Gaussian Processes
and Thompson sampling for exploration/exploitation trade-off.
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

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube


# ===== TURBO-1 Single Trust Region Optimizer ===== #


class Turbo1:
    """
    Single-trust-region Bayesian optimization using TURBO algorithm.

    Optimizes continuous functions using a Gaussian Process surrogate with
    Thompson sampling for candidate generation within an adaptive trust region.
    Automatically expands or shrinks the trust region based on optimization
    progress.

    Initialization Parameters:
    --------------------------
    f : callable
        Objective function to minimize. Takes numpy array of shape (d,) and returns float.
    lb : numpy array
        Lower bounds for each dimension, shape (d,).
    ub : numpy array
        Upper bounds for each dimension, shape (d,).
    n_init : int
        Number of initial Latin hypercube design points (2*dim recommended).
    max_evals : int
        Total evaluation budget limit.
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
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    ):

        # basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub >= lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # tolerances and counters
        self.n_cand = min(100 * self.dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # initialize parameters
        self._restart()

    def _restart(self):
        """
        Reinitialize local trust region state for new optimization phase.

        Resets all local state variables tracking the current trust region including:
        - Local design point history (_X and _fX)
        - Success and failure counters
        - Trust region length parameter

        This is called at initialization and when starting fresh trust region phases.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        """
        Expand or contract trust region length based on optimization progress.

        Implements adaptive trust region sizing:
        - Expansion: If succtol consecutive improvements found, double region length
        - Contraction: If failtol consecutive failures, halve region length
        - Improvement criterion: fX_next < min(fX) - 1e-3 * |min(fX)|

        Parameters
        ----------
        fX_next : np.ndarray
            Function evaluation results from candidate batch. Shape (batch_size, 1).

        Returns
        -------
        None
        """
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:
            self.length /= 2.0
            self.failcount = 0


    # ==== Candidate Generation and Optimization ==== #


    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """
        Generate candidate points within trust region using GP and Thompson sampling.

        This method implements the core candidate generation pipeline:
        1. Standardizes function values using median/std normalization
        2. Trains GP on standardized data using GPyTorch with ARD kernels
        3. Computes trust region boundaries from GP kernel lengthscales
        4. Generates candidate pool via Sobol sequences within trust region
        5. Applies perturbation mask for diversity
        6. Samples candidates using GP posterior with Thompson sampling

        The method assumes X has been scaled to [0,1]^d. It generates candidates
        by sampling from the GP posterior at multiple sample paths (batch_size),
        which encourages diversity and exploration.

        Parameters
        ----------
        X : np.ndarray
            Historical design points scaled to [0,1]^d. Shape (n_evals, dim).
        fX : np.ndarray
            Function values at X. Shape (n_evals, 1).
        length : float
            Trust region length parameter (typically in [0.5^7, 1.6]).
        n_training_steps : int
            Number of ADAM optimization steps for hyperparameter tuning.
        hypers : dict
            Previous GP hyperparameters. Keys: 'mean', 'signal_var', 'noise_var', 'lengthscales'.

        Returns
        -------
        X_cand : np.ndarray
            Candidate points in [0,1]^d. Shape (n_cand, dim).
        y_cand : np.ndarray
            Sampled values at candidates from GP. Shape (n_cand, batch_size).
        hypers : dict
            Updated GP hyperparameters after training.
        """
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            hypers = gp.state_dict()

        # create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        gp = gp.to(dtype=dtype, device=device)

        # use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers


    # ==== Candidate Selection ==== #


    def _select_candidates(self, X_cand, y_cand):
        """
        Select best candidate from pool for each batch dimension.

        Implements greedy batch point selection: for each batch slot, selects the
        candidate with minimum predicted value, then masks that point to prevent
        duplicate selection in subsequent batch slots.

        Parameters
        ----------
        X_cand : np.ndarray
            Candidate pool of points. Shape (n_cand, dim).
        y_cand : np.ndarray
            Predicted values at candidates for each batch slot. Shape (n_cand, batch_size).
            Each column represents predictions for one batch slot.

        Returns
        -------
        X_next : np.ndarray
            Selected points for evaluation. Shape (batch_size, dim).
        """
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
        return X_next


    # ===== Main Optimization Loop ===== #


    def optimize(self):
        """
        Run the complete trust region Bayesian optimization process.

        Implements the main TURBO-1 optimization loop:
        1. Iteratively restarts trust region with new random initial points
        2. Generates initial design via Latin hypercube sampling
        3. Performs Thompson sampling to suggest next batch of candidates
        4. Evaluates batch and updates trust region size based on progress
        5. Tracks global and local optimization history
        6. Terminates when budget exhausted or trust region shrinks below threshold

        The optimization alternates between:
        - TR phases: Exploit with GP-guided sampling within adaptive trust region
        - Restarts: Refresh with new random design when TR shrinks too much

        Global optimization history (self.X, self.fX) accumulates all evaluations.
        Local history (self._X, self._fX) tracks current trust region phase.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Updates optimizer state attributes in-place:
        - self.X: All evaluated points
        - self.fX: Function values at all points
        - self.n_evals: Total evaluations used
        - self._X, self._fX: Current trust region data
        - self.length: Current trust region size
        """
        if len(self._fX) > 0 and self.verbose:
            n_evals, fbest = self.n_evals, self._fX.min()
            print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
            sys.stdout.flush()

        # initialize parameters
        self._restart()

        # generate and evalute initial design points
        X_init = latin_hypercube(self.n_init, self.dim)
        X_init = from_unit_cube(X_init, self.lb, self.ub)
        fX_init = np.array([[self.f(x)] for x in X_init])

        # update budget and set as initial data for this TR
        self.n_evals += self.n_init
        self._X = deepcopy(X_init)
        self._fX = deepcopy(fX_init)

        # append data to the global history
        self.X = np.vstack((self.X, deepcopy(X_init)))
        self.fX = np.vstack((self.fX, deepcopy(fX_init)))

        if self.verbose:
            fbest = self._fX.min()
            print(f"Starting from fbest = {fbest:.4}")
            sys.stdout.flush()

        # Thompson sample to get next suggestions
        while self.n_evals < self.max_evals and self.length >= self.length_min:
            X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

            fX = deepcopy(self._fX).ravel()

            X_cand, y_cand, _ = self._create_candidates(
                X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
            )
            X_next = self._select_candidates(X_cand, y_cand)

            X_next = from_unit_cube(X_next, self.lb, self.ub)

            fX_next = np.array([[self.f(x)] for x in X_next])

            self._adjust_length(fX_next)

            self.n_evals += self.batch_size
            self._X = np.vstack((self._X, X_next))
            self._fX = np.vstack((self._fX, fX_next))

            if self.verbose and fX_next.min() < self.fX.min():
                n_evals, fbest = self.n_evals, fX_next.min()
                print(f"{n_evals}) New best: {fbest:.4}")
                sys.stdout.flush()

            self.X = np.vstack((self.X, deepcopy(X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))
