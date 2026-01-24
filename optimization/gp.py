"""
gp.py

Authours: dkochar, natelgrw
Last Edited: 01/15/2026

Gaussian Process model for Bayesian optimization in the TURBO framework.
Implements exact GP inference using GPyTorch for efficient kernel learning.
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
import torch

from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP


# ===== Gaussian Process Model ===== #


class GP(ExactGP):
    """
    Exact Gaussian Process using Matern kernel with optional ARD.
    
    Implements a standard GP with a scaled Matern 5/2 kernel for use
    as a surrogate model in Bayesian optimization. Supports Automatic
    Relevance Determination (ARD) for adaptive length scales.
    
    Initialization Parameters:
    --------------------------
    train_x : torch tensor
        Training input points, shape (n_train, d).
    train_y : torch tensor
        Training target values, shape (n_train,).
    likelihood : GaussianLikelihood
        Likelihood model for GP inference.
    lengthscale_constraint : Interval constraint
        Bounds for kernel length scales.
    outputscale_constraint : Interval constraint
        Bounds for kernel output scale.
    ard_dims : int or None
        Number of dimensions for ARD. If None, uses single global length scale.
    """

    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims):
    
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        
        # Matern 5/2 kernel with optional ARD
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=2.5)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        """
        Forward pass through GP to compute mean and covariance.

        Parameters:
        -----------
        x : torch tensor
            Input points, shape (n_points, d).
        
        Returns:
        --------
        MultivariateNormal
            Predictive distribution with mean and covariance.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# ===== GP Training ===== #


def train_gp(train_x, train_y, use_ard, num_steps, hypers={}):
    """
    Fit a Gaussian Process to training data.

    Trains a GP model on standardized training data in [0, 1]^d using
    maximum likelihood estimation via ADAM optimization. Supports both
    global and per-dimension (ARD) length scales.

    Parameters:
    -----------
    train_x : numpy array or torch tensor
        Training input points, shape (n_train, d), values in [0, 1]^d.
    train_y : numpy array or torch tensor
        Training target values, shape (n_train,), standardized.
    use_ard : bool
        If True, use Automatic Relevance Determination (one length scale per dimension).
        If False, use single global length scale.
    num_steps : int
        Number of ADAM optimization steps for hyperparameter learning.
    hypers : dict, optional
        Initial hyperparameter values. If empty, uses smart defaults.
    
    Returns:
    --------
    GP
        Fitted Gaussian Process model in evaluation mode.
    
    Raises:
    -------
    AssertionError
        If input dimensions are invalid or data shapes mismatch.
    """
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # === Define Hyperparameter Constraints === #

    noise_constraint = Interval(5e-4, 0.2)
    
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
    
    outputscale_constraint = Interval(0.05, 20.0)

    # === Initialize GP Model === #

    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    
    ard_dims = train_x.shape[1] if use_ard else None
    
    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # === Optimize Hyperparameters === #

    model.train()
    likelihood.train()

    mll = ExactMarginalLogLikelihood(likelihood, model)

    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = 0.5
        hypers["likelihood.noise"] = 0.005
        model.initialize(**hypers)

    # ADAM optimizer for hyperparameter learning
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

    # optimization loop
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model
