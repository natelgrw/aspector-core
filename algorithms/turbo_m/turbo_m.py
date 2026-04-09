"""
turbo_m.py

Author: natelgrw, dkochar
Last Edited: 02/13/2026

TuRBO-M algorithm implementation using BoTorch and GPyTorch. Uses robust standardized scalarization for 
multi-objective handling, and dynamic resource allocation across 10 supported trust region personas. 
"""

import math
import torch
import numpy as np
from dataclasses import dataclass
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
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
        """
        Initializes the TurboState. If failure_tolerance is not provided, it defaults to 2 * dim.
        """
        if math.isnan(self.failure_tolerance):
            self.failure_tolerance = 2 * self.dim

class TurboMSizingGenerator:

    _SPEC_STD_FLOORS = {
        'gain_ol_dc_db': 1.0,
        'pm_deg': 1.0,
        'cmrr_dc_db': 1.0,
        'psrr_dc_db': 1.0,
        'ugbw_hz': 1e6,
        'slew_rate_v_us': 1.0,
        'output_voltage_swing_range_v': 1e-3,
        'power_w': 1e-6,
        'integrated_noise_vrms': 1e-9,
        'thd_db': 0.1,
        'vos_v': 1e-4,
        'settle_time_small_ns': 1e-3,
        'settle_time_large_ns': 1e-3,
        'estimated_area_um2': 0.1,
    }

    def __init__(self, 
                 dim, 
                 specs_weights=None,
                 num_trust_regions=5, 
                 batch_size=64, 
                 failure_tolerance=None, 
                 device=None, 
                 dtype=torch.double,
                 **kwargs):
        """
        TurboMSizingGenerator: State-of-the-Art TuRBO implementation for Analog Optimization.

        Parameters:
        -----------
        dim (int): Dimensionality of the design space.
        specs_weights (dict): Optional dictionary of weights for scalarization.
        num_trust_regions (int): Number of parallel trust regions to maintain.
        batch_size (int): Total number of candidates to generate per iteration (split across TRs
        failure_tolerance (int): Number of consecutive failures before shrinking a TR. Defaults to 2*dim.
        device (torch.device): Device for computations. Defaults to CUDA if available.
        dtype (torch.dtype): Data type for tensors. Defaults to torch.double for GP stability.
        """
        self.dim = dim
        self.num_trust_regions = num_trust_regions
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # initialize trust regions
        self.state = [
            TurboState(dim=dim, batch_size=batch_size)
            for _ in range(num_trust_regions)
        ]
        if failure_tolerance is not None:
            for s in self.state:
                s.failure_tolerance = failure_tolerance

        # data storage
        self.X = torch.empty((0, dim), dtype=dtype, device=self.device)
        self.Y = torch.empty((0, 1), dtype=dtype, device=self.device)
        
        if specs_weights is None:
            raise ValueError("specs_weights must be provided explicitly.")

        self.spec_stats = {}
        self.weights = specs_weights

    def _spec_std_floor(self, key):
        """
        Return a conservative per-spec standard-deviation floor.
        """
        return float(self._SPEC_STD_FLOORS.get(key, 1.0))

    def load_state(self, X_init, Y_init=None):
        """
        Warm-start or resume the optimizer with existing data or full state dict.
        Accepts either (X_init, Y_init) for warm start, or a dict for full resume.

        Parameters:
        -----------
        X_init (torch.Tensor or np.ndarray or dict): Initial design points or full state dict
        Y_init (torch.Tensor or np.ndarray): Initial scalarized values (optional if X_init is a state dict)
        """
        if isinstance(X_init, dict):
            # full resume from state dict
            state = X_init
            self.state = state['state']
            self.X = state['X']
            self.Y = state['Y']
            self.spec_stats = state.get('spec_stats', {})
            self.weights = state.get('weights', self.weights)
            return

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
                
    def _update_spec_stats(self, specs_list):
        """
        Update running statistics for normalization. 
        Only looks at valid designs to avoid skewing stats with broken simulation artifacts.

        Parameters:
        -----------
        specs_list (list of dict): List of specification dictionaries from recent evaluations.
        """
        for key in self.weights.keys():
            if key not in self.spec_stats:
                self.spec_stats[key] = {'vals': []}
            
            vals = []
            for s in specs_list:
                if s and s.get('valid', False) and s.get(key) is not None:
                    v = s.get(key)
                    if isinstance(v, (list, tuple)): v = abs(v[1]-v[0])
                    vals.append(float(v))
            
            if vals:
                # append to history
                self.spec_stats[key]['vals'].extend(vals)
                
                # compute robust stats
                data = torch.tensor(self.spec_stats[key]['vals'])
                self.spec_stats[key]['mean'] = data.mean().item()
                std_raw = data.std().item()
                self.spec_stats[key]['std'] = max(std_raw, self._spec_std_floor(key))

    def scalarize_specs(self, specs_list, update_stats=True):
        """
        Robust Standardized Scalarization.
        Converts all metrics to Z-scores before weighting.

        Parameters:
        -----------
        specs_list (list of dict): List of specification dictionaries from recent evaluations.
        update_stats (bool): Whether to update running statistics with this batch before scalarization.

        Returns:
        --------
        torch.Tensor: Scalarized cost values for each design in the batch.
        """
        if update_stats:
            self._update_spec_stats(specs_list)
            
        y_vals = []

        # hard constraint
        for specs in specs_list:
            if specs is None or not specs.get('valid', False):
                y_vals.append([1e6])
                continue

            cost = 0.0
            reward = 1.0

            pm_deg = specs.get('pm_deg', 0.0)
            if pm_deg < 45.0:
                reward *= math.exp(-0.1 * (45.0 - pm_deg))

            gain = specs.get('gain_ol_dc_db', 0.0)
            if gain < 35.0:
                reward *= math.exp(-0.2 * (35.0 - gain))

            ugbw_hz = specs.get('ugbw_hz', 0.0)
            if ugbw_hz < 1e6:
                ugbw_mhz = ugbw_hz / 1e6
                reward *= math.exp(-2.0 * (1.0 - ugbw_mhz))

            cost += -math.log(max(reward, 1e-10)) * 50.0

            # normalized objectives
            pm_target = self.weights.get('_pm_deg_target', None)
            pm_range = self.weights.get('_pm_deg_range', 0.0)

            for key, weight in self.weights.items():
                if key.startswith('_'): 
                    continue

                if key == 'pm_deg':
                    continue 

                raw_val = specs.get(key, 0.0)
                if isinstance(raw_val, (list, tuple)): raw_val = abs(raw_val[1] - raw_val[0])
                if raw_val is None: raw_val = 0.0

                # retrieve stats and z-score
                mu = self.spec_stats[key].get('mean', 0.0)
                sigma = self.spec_stats[key].get('std', self._spec_std_floor(key))
                sigma = max(float(sigma), self._spec_std_floor(key))

                z_val = (raw_val - mu) / sigma

                if key in ['gain_ol_dc_db', 'ugbw_hz', 'cmrr_dc_db', 'psrr_dc_db', 'slew_rate_v_us', 'output_voltage_swing_range_v']:
                    # maximize -> minimize negative Z
                    term = -1.0 * z_val
                else:
                    # minimize -> minimize positive Z
                    term = 1.0 * z_val

                cost += weight * term

            # optional targeted metrics
            if pm_target is not None and pm_range is not None and pm_target > 0:
                diff = abs(pm_deg - pm_target)
                if diff > pm_range:
                    cost += self.weights.get('pm_deg', 0.0) * (diff - pm_range)

            y_vals.append([cost])
            
        return torch.tensor(y_vals, dtype=self.dtype, device=self.device)

    def tell(self, X_new, specs_new):
        """
        Ingest new batch of results.
        Automatically associates points with the correct Trust Region to update state.

        Parameters:
        -----------
        X_new (torch.Tensor or np.ndarray): New design points evaluated.
        specs_new (list of dict): Corresponding specifications for each design point.
        """
        if not isinstance(X_new, torch.Tensor):
            X_new = torch.tensor(X_new, dtype=self.dtype, device=self.device)
        
        # scalarize
        Y_new = self.scalarize_specs(specs_new, update_stats=True)
        
        # append to Database
        self.X = torch.cat([self.X, X_new], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)
        
        for state in self.state:
            if state.restart_triggered: continue
            
            if state.center_idx >= 0 and state.center_idx < len(self.X):
                center = self.X[state.center_idx].unsqueeze(0)
            else:
                continue
            
            # use L-infinity distance so ownership matches TR hypercube geometry
            dists = torch.max(torch.abs(X_new - center), dim=-1)[0]

            # point must lie inside current TR hypercube (with tiny numerical slack)
            mask = dists <= (state.length / 2.0 + 1e-5)
            
            relevant_indices = torch.where(mask)[0]
            
            if len(relevant_indices) > 0:
                best_in_batch = Y_new[relevant_indices].min().item()
                
                if best_in_batch < state.best_value - 1e-4:
                    state.success_counter += 1
                    state.failure_counter = 0
                    state.best_value = best_in_batch

                else:
                    state.success_counter = 0
                    state.failure_counter += 1
            else:
                pass

            # update hypers
            if state.success_counter >= state.success_tolerance:
                state.length = min(2.0 * state.length, state.length_max)
                state.success_counter = 0
            elif state.failure_counter >= state.failure_tolerance:
                state.length /= 2.0
                state.failure_counter = 0
                
            if state.length < state.length_min:
                state.restart_triggered = True

    def ask(self, n_samples=None, context_u=None, context_dim=0):
        """
        Alias for generate_batch, matching standard Optimizer API.

        Parameters:
        -----------
        n_samples (int): Optional override for batch size for this ask. If None, uses default batch size.
        context_u (torch.Tensor | np.ndarray | None): Optional context vectors
            of shape (batch_size, context_dim) to condition candidate generation.
        context_dim (int): Number of leading dimensions treated as context.
        """
        if n_samples is not None:
            self.batch_size = n_samples

        if context_u is not None:
            if not isinstance(context_u, torch.Tensor):
                context_u = torch.tensor(context_u, dtype=self.dtype, device=self.device)
            else:
                context_u = context_u.to(device=self.device, dtype=self.dtype)
            if context_u.ndim != 2:
                raise ValueError("context_u must be 2D [batch_size, context_dim]")
            if context_u.shape[0] != self.batch_size:
                raise ValueError("context_u first dimension must equal batch size")
            if context_dim <= 0 or context_u.shape[1] != context_dim:
                raise ValueError("context_dim must be >0 and match context_u.shape[1]")
             
        return self.generate_batch(context_u=context_u, context_dim=context_dim)

    def generate_batch(self, n_candidates=2000, context_u=None, context_dim=0):
        """
        Generate candidates with Dynamic Resource Allocation.

        Parameters:
        -----------
        n_candidates (int): Number of candidate points to sample for Thompson Sampling in each TR.
        """
        X_next = torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
        
        # dynamic allocation algorithm
        scores = np.array([1.0 / (1.0 + s.failure_counter) for s in self.state])
        probs = scores / scores.sum()
        
        # discrete allocation of batch slots
        counts = np.random.multinomial(self.batch_size, probs)
        
        ctx_offset = 0

        for i, state in enumerate(self.state):
            n_samples = counts[i]
            if n_samples == 0: continue

            ctx_chunk = None
            if context_u is not None and context_dim > 0:
                ctx_chunk = context_u[ctx_offset:ctx_offset + n_samples]
                ctx_offset += n_samples
            
            if state.restart_triggered:
                # reset
                state.length = 0.8
                state.failure_counter = 0
                state.success_counter = 0
                state.best_value = float('inf')
                state.restart_triggered = False
                
                # sobol restart
                sobol = SobolEngine(self.dim, scramble=True)
                cand = sobol.draw(n_samples).to(dtype=self.dtype, device=self.device)
                if ctx_chunk is not None:
                    cand[:, :context_dim] = ctx_chunk
                X_next = torch.cat([X_next, cand], dim=0)
                
                continue
            
            # determine trust region center and fit local model
            
            if len(self.X) > 0:
                if state.center_idx == -1:
                    state.center_idx = self.Y.argmin().item()

                old_center = self.X[state.center_idx].unsqueeze(0)
                dists = torch.cdist(self.X, old_center).squeeze()
                mask = dists <= state.length * 2.0
                
                if mask.any():
                    subset_y = self.Y[mask]
                    subset_indices = torch.where(mask)[0]
                    local_best_local_idx = subset_y.argmin()
                    state.center_idx = subset_indices[local_best_local_idx].item()
                    
                x_center = self.X[state.center_idx].unsqueeze(0)
                
                # model fitting
                dists = torch.cdist(self.X, x_center).squeeze()
                
                n_train = min(len(self.X), 256)
                _, indices = torch.topk(dists, n_train, largest=False)
                
                train_X = self.X[indices]
                train_Y = self.Y[indices]
                
                y_std = train_Y.std()
                if torch.isnan(y_std) or y_std == 0.0:
                    y_std = 1.0
                train_Y_std = (train_Y - train_Y.mean()) / (y_std + 1e-6)
                
                covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_prior=GammaPrior(3.0, 6.0)))
                gp = SingleTaskGP(train_X, train_Y_std, covar_module=covar)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                try:
                    fit_gpytorch_model(mll)
                except:
                    pass

                # switch to eval mode before posterior sampling
                gp.eval()
                gp.likelihood.eval()
                
                # Thompson sampling in trust region
                with torch.no_grad():
                    sobol = SobolEngine(self.dim, scramble=True)
                    pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
                    
                    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
                    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

                    if ctx_chunk is not None:
                        selected = []
                        for j in range(n_samples):
                            cand_X = tr_lb + (tr_ub - tr_lb) * pert
                            cand_X[:, :context_dim] = ctx_chunk[j].unsqueeze(0).expand(n_candidates, -1)
                            posterior = gp.posterior(cand_X)
                            samples = posterior.rsample(sample_shape=torch.Size([1])).view(-1)
                            best_idx = torch.argmin(samples)
                            selected.append(cand_X[best_idx].unsqueeze(0))
                        X_tr = torch.cat(selected, dim=0) if selected else torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
                    else:
                        cand_X = tr_lb + (tr_ub - tr_lb) * pert
                        posterior = gp.posterior(cand_X)
                        samples = posterior.rsample(sample_shape=torch.Size([1])).view(-1)
                        _, best_idxs = torch.topk(samples, n_samples, largest=False)
                        X_tr = cand_X[best_idxs]
                    
                    X_next = torch.cat([X_next, X_tr], dim=0)
                    
            else:
                # cold start with Sobol if no data yet
                sobol = SobolEngine(self.dim, scramble=True)
                cold = sobol.draw(n_samples).to(dtype=self.dtype, device=self.device)
                if ctx_chunk is not None:
                    cold[:, :context_dim] = ctx_chunk
                X_next = torch.cat([X_next, cold], dim=0)

        return X_next
