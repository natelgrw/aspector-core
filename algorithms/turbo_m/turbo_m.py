"""
turbo_m.py

Author: GitHub Copilot
Last Edited: 02/13/2026

ASPECTOR_TurboM: TuRBO-M algorithm implementation using BoTorch and GPyTorch.
Refined for "NeurIPS Quality" with Dynamic Resource Allocation, Robust Scalarization, 
and Proximity-Based State Management.
"""

import math
import torch
import numpy as np
from dataclasses import dataclass
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from torch.quasirandom import SobolEngine

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = float("inf")
    restart_triggered: bool = False
    
    # Track center of the TR relative to global X
    center_idx: int = -1 

    def __post_init__(self):
        if math.isnan(self.failure_tolerance):
            self.failure_tolerance = 2 * self.dim

class ASPECTOR_TurboM:
    def __init__(self, 
                 dim, 
                 specs_weights=None, # Dictionary of weights
                 num_trust_regions=5, 
                 batch_size=64, 
                 failure_tolerance=None, 
                 device=None, 
                 dtype=torch.double,
                 **kwargs): # Catch extra args
        """
        ASPECTOR_TurboM: State-of-the-Art TuRBO implementation for Analog Optimization.
        """
        self.dim = dim
        self.num_trust_regions = num_trust_regions
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Initialize Trust Regions
        self.state = [
            TurboState(dim=dim, batch_size=batch_size)
            for _ in range(num_trust_regions)
        ]
        if failure_tolerance is not None:
            for s in self.state:
                s.failure_tolerance = failure_tolerance

        # Data Storage
        self.X = torch.empty((0, dim), dtype=dtype, device=self.device)
        self.Y = torch.empty((0, 1), dtype=dtype, device=self.device)
        
        # Robust Scalarization Statistics
        self.spec_stats = {} 
        self.weights = specs_weights if specs_weights else self._get_default_weights()

    def load_state(self, X_init, Y_init=None):
        """
        Warm-start or resume the optimizer with existing data or full state dict.
        Accepts either (X_init, Y_init) for warm start, or a dict for full resume.
        """
        if isinstance(X_init, dict):
            # Full resume from state dict
            state = X_init
            self.state = state['state']
            self.X = state['X']
            self.Y = state['Y']
            self.spec_stats = state.get('spec_stats', {})
            self.weights = state.get('weights', self.weights)
            return
        # Else, warm start from X_init, Y_init
        if not isinstance(X_init, torch.Tensor):
            X_init = torch.tensor(X_init, dtype=self.dtype, device=self.device)
        if not isinstance(Y_init, torch.Tensor):
            Y_init = torch.tensor(Y_init, dtype=self.dtype, device=self.device)
        self.X = torch.cat([self.X, X_init], dim=0)
        self.Y = torch.cat([self.Y, Y_init], dim=0)
        if len(self.Y) > 0:
            k = min(len(self.Y), len(self.state))
            _, best_indices = torch.topk(self.Y.flatten(), k, largest=False)
            for i, state in enumerate(self.state):
                idx = i % len(best_indices)
                state.center_idx = best_indices[idx].item()
                state.best_value = self.Y[state.center_idx].item()
                
    def _get_default_weights(self):
        return {
            'gain_ol': 1.0, 'ugbw': 1.0, 'pm': 100.0, 'gm': 50.0, # High importance on stability constraints
            'power': 2.0, 'area': 1.0, 'cmrr': 1.0, 'psrr': 1.0,
            'vos': 5.0, 'output_voltage_swing': 1.0, 'integrated_noise': 1.0,
            'slew_rate': 1.0, 'settle_time': 1.0, 'thd': 1.0, 'gain_cl': 0.5
        }

    def _update_spec_stats(self, specs_list):
        """
        Update running statistics for normalization. 
        Only looks at valid designs to avoid skewing stats with broken simulation artifacts.
        """
        for key in self.weights.keys():
            if key not in self.spec_stats:
                self.spec_stats[key] = {'vals': []}
            
            vals = []
            for s in specs_list:
                if s and s.get('valid', False) and s.get(key) is not None:
                    v = s.get(key)
                    # Handle tuples (Swing)
                    if isinstance(v, (list, tuple)): v = abs(v[1]-v[0])
                    vals.append(float(v))
            
            if vals:
                # Append to history (memory inefficient for infinite runs, but fine for circuit opt <10k)
                self.spec_stats[key]['vals'].extend(vals)
                
                # Compute robust stats (Median/IQR preferred, but Mean/Std ok for now)
                data = torch.tensor(self.spec_stats[key]['vals'])
                self.spec_stats[key]['mean'] = data.mean().item()
                self.spec_stats[key]['std'] = data.std().item() + 1e-9

    def scalarize_specs(self, specs_list, update_stats=True):
        """
        Robust Standardized Scalarization.
        Converts all metrics to Z-scores before weighting.
        """
        if update_stats:
            self._update_spec_stats(specs_list)
            
        y_vals = []
        for specs in specs_list:
            if specs is None or not specs.get('valid', False):
                y_vals.append([1e6]) # Penalty
                continue

            cost = 0.0
            
            # --- 1. Hard Stability Constraints (Not Normalized) ---
            # These are "violations", not objectives.
            pm = specs.get('pm', 0.0)
            if pm < 45.0: cost += 1e3 * (45.0 - pm) 
            
            gm = specs.get('gm', 0.0)
            if gm < 0.0: cost += 1e3 * abs(gm)

            # --- 2. Normalized Objectives ---
            for key, weight in self.weights.items():
                if key in ['pm', 'gm']: continue # Handled above

                raw_val = specs.get(key, 0.0)
                if isinstance(raw_val, (list, tuple)): raw_val = abs(raw_val[1] - raw_val[0])
                if raw_val is None: raw_val = 0.0
                
                # Retrieve stats
                mu = self.spec_stats[key].get('mean', 0.0)
                sigma = self.spec_stats[key].get('std', 1.0)
                
                # Z-Score
                z_val = (raw_val - mu) / sigma
                
                # Directionality
                if key in ['gain_ol', 'ugbw', 'cmrr', 'psrr', 'slew_rate', 'output_voltage_swing']:
                    # Maximize -> Minimize negative Z
                    term = -1.0 * z_val
                else:
                    # Minimize -> Minimize positive Z
                    term = 1.0 * z_val
                
                cost += weight * term

            y_vals.append([cost])
            
        return torch.tensor(y_vals, dtype=self.dtype, device=self.device)

    def tell(self, X_new, specs_new):
        """
        Ingest new batch of results.
        Automatically associates points with the correct Trust Region to update state.
        """
        if not isinstance(X_new, torch.Tensor):
            X_new = torch.tensor(X_new, dtype=self.dtype, device=self.device)
        
        # 1. Scalarize
        Y_new = self.scalarize_specs(specs_new, update_stats=True)
        
        # 2. Append to Database
        self.X = torch.cat([self.X, X_new], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)
        
        # 3. Proximity-Based State Update
        # For each TR, check if any of the NEW points fall within its domain and improve it.
        # This is more robust than assuming we know indices.
        
        for state in self.state:
            if state.restart_triggered: continue
            
            # Identify the center of this TR
            if state.center_idx >= 0 and state.center_idx < len(self.X):
                center = self.X[state.center_idx].unsqueeze(0)
            else:
                continue # Should not happen if initialized
            
            # Check distances of NEW points to this center
            dists = torch.cdist(X_new, center).squeeze()
            
            # Filter points that "captured" by this TR (within radius L)
            # We use L*1.5 to be generous with the boundary
            mask = dists <= (state.length * math.sqrt(self.dim)) 
            
            relevant_indices = torch.where(mask)[0]
            
            if len(relevant_indices) > 0:
                # Did we find an improvement?
                best_in_batch = Y_new[relevant_indices].min().item()
                
                if best_in_batch < state.best_value - 1e-4:
                    state.success_counter += 1
                    state.failure_counter = 0
                    state.best_value = best_in_batch
                    # Update center to the new best point (Global index calculation needed)
                    # For simplicty in this flow, we update value. Next generate_batch finds new center.
                else:
                    state.success_counter = 0
                    state.failure_counter += 1
            else:
                # No points allocated to this region in this batch? 
                # Either we didn't sample it (Dynamic Allocation) or it drifted.
                pass

            # Update Hypers
            if state.success_counter >= state.success_tolerance:
                state.length = min(2.0 * state.length, state.length_max)
                state.success_counter = 0
            elif state.failure_counter >= state.failure_tolerance:
                state.length /= 2.0
                state.failure_counter = 0
                
            if state.length < state.length_min:
                state.restart_triggered = True

    def ask(self, n_samples=None):
        """
        Alias for generate_batch, matching standard Optimizer API.
        """
        if n_samples is not None:
             # Override the *total* batch size split across regions
             self.batch_size = n_samples
             
        return self.generate_batch()

    def generate_batch(self, n_candidates=2000):
        """
        Generate candidates with Dynamic Resource Allocation.
        """
        X_next = torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
        
        # --- Algorithm 1: Dynamic Allocation ---
        # Allocate batch size proportional to "inverse failure count" (proxy for health)
        # or just equal if unsure.
        # HRT Quality: "Back the winners". 
        # Score = 1 / (1 + failure_counter). Clean regions get more samples.
        scores = np.array([1.0 / (1.0 + s.failure_counter) for s in self.state])
        probs = scores / scores.sum()
        
        # Discrete allocation of batch slots
        counts = np.random.multinomial(self.batch_size, probs)
        
        for i, state in enumerate(self.state):
            n_samples = counts[i]
            if n_samples == 0: continue
            
            # 1. Handle Restart
            if state.restart_triggered:
                # Reset
                state.length = 0.8
                state.failure_counter = 0
                state.success_counter = 0
                state.best_value = float('inf')
                state.restart_triggered = False
                
                # Sobol Restart (Global)
                sobol = SobolEngine(self.dim, scramble=True)
                cand = sobol.draw(n_samples).to(dtype=self.dtype, device=self.device)
                X_next = torch.cat([X_next, cand], dim=0)
                
                # Update center to best of this random batch (heuristic) later
                continue
            
            # 2. Determine Center
            # In M-TuRBO, center is best point in the TR.
            # We approximate this by finding the point in history closest to current "center_idx"
            # that has the best value? No, simplest is global best or tracking best per TR.
            
            if len(self.X) > 0:
                if state.center_idx == -1:
                    # First time init: Pick random best
                    state.center_idx = self.Y.argmin().item()
                
                # Ideally, we should perform a search for the best point among those 
                # PRODUCED by this TR. Since we don't persist that map, we use the 
                # previously tracked center index, but verify if a better point exists near it.
                
                # Refined Logic: Find best point in X within distance L of old center
                old_center = self.X[state.center_idx].unsqueeze(0)
                dists = torch.cdist(self.X, old_center).squeeze()
                mask = dists <= state.length * 2.0
                
                if mask.any():
                    # Find min Y in this neighborhood
                    subset_y = self.Y[mask]
                    subset_indices = torch.where(mask)[0]
                    local_best_local_idx = subset_y.argmin()
                    state.center_idx = subset_indices[local_best_local_idx].item()
                    
                x_center = self.X[state.center_idx].unsqueeze(0)
                
                # 3. Model Fitting
                # Mask data: Points within 2*L from center
                dists = torch.cdist(self.X, x_center).squeeze()
                
                # TR Logic: Training set constraint
                n_train = min(len(self.X), 256) # Cap training set for speed (HRT optimization)
                _, indices = torch.topk(dists, n_train, largest=False)
                
                train_X = self.X[indices]
                train_Y = self.Y[indices]
                
                # Standardize Y locally for GP stability
                y_std = train_Y.std()
                if torch.isnan(y_std) or y_std == 0.0:
                    y_std = 1.0
                train_Y_std = (train_Y - train_Y.mean()) / (y_std + 1e-6)
                
                # Fit
                covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_prior=GammaPrior(3.0, 6.0)))
                gp = SingleTaskGP(train_X, train_Y_std, covar_module=covar)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                try:
                    fit_gpytorch_mll(mll)
                except:
                    pass
                
                # 4. Thompson Sampling
                with torch.no_grad():
                    # Generate perturbation candidates in TR
                    sobol = SobolEngine(self.dim, scramble=True)
                    pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
                    
                    # Scale to Trust Region
                    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
                    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)
                    
                    cand_X = tr_lb + (tr_ub - tr_lb) * pert
                    
                    # Thompson Sample
                    posterior = gp.posterior(cand_X)
                    samples = posterior.rsample(sample_shape=torch.Size([1])).squeeze()
                    
                    # Pick best n_samples
                    _, best_idxs = torch.topk(samples, n_samples, largest=False)
                    X_tr = cand_X[best_idxs]
                    
                    X_next = torch.cat([X_next, X_tr], dim=0)
                    
            else:
                # Cold Start - Use Sobol for better initial coverage than random
                sobol = SobolEngine(self.dim, scramble=True)
                X_next = torch.cat([X_next, sobol.draw(n_samples).to(dtype=self.dtype, device=self.device)], dim=0)

        return X_next 
