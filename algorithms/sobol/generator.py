"""
generator.py

Author: GitHub Copilot
Last Edited: 02/05/2026

Sobol sequence generator for circuit sizing parameters.
Generates valid design points respecting globalsy constraints and technology rules.
"""

import numpy as np
from scipy.stats import qmc
import math
import sys
import os

# Add project root to path to import globalsy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from simulator import globalsy

# Approximate technology constants since we cannot access the models directly
# These drive the dependent constraints like L_min and Vdd_nominal
TECH_CONSTANTS = {
    7:  {'lmin': 7e-9,  'vdd_nom': 0.70},
    10: {'lmin': 10e-9, 'vdd_nom': 0.75},
    14: {'lmin': 14e-9, 'vdd_nom': 0.80},
    16: {'lmin': 16e-9, 'vdd_nom': 0.80},
    20: {'lmin': 20e-9, 'vdd_nom': 0.90},
}

class SobolSizingGenerator:
    def __init__(self, sizing_params_list, seed=None):
        """
        Initialize the generator.

        Args:
            sizing_params_list (list): List of parameter names found in the netlist (e.g. ['nA1', 'nB1', ...])
            seed (int, optional): Random seed for reproducibility.
        """
        self.sizing_params = sizing_params_list
        self.seed = seed
        self.tech_nodes = globalsy.testbench_params['Fet_num']
        
        # Categorize sizing parameters to determine dimensionality
        # Fixed Environment/Testbench params that act as inputs to the simulation
        self.fixed_params = [
            'fet_num', 'vdd', 'vcm', 'tempc', 
            'rfeedback_val', 'rsrc_val', 'cload_val'
        ]
        
        # Total dimensions = Fixed params + Sizing params
        self.dim = len(self.fixed_params) + len(self.sizing_params)
        
        # Initialize Scipy Sobol engine
        # scramble=True is generally recommended for better uniformity in lower sample counts
        self.engine = qmc.Sobol(d=self.dim, scramble=True, seed=seed)

    def generate(self, n_samples):
        """
        Generate n_samples of valid design configurations.

        Args:
            n_samples (int): Number of samples to generate.
                             Note: Sobol sequences work best with powers of 2. 
                             The engine may generate the next power of 2 internally or we limit output.

        Returns:
            list[dict]: List of dictionaries, each containing a full set of parameters.
        """
        
        # Draw samples from the Unit Hypercube [0, 1]^d
        # m = ceil(log2(n_samples)) if strict power of 2 is needed, 
        # but qmc.Sobol.random(n) works for any n (just continues sequence)
        
        # We find the next power of 2 to ensure balance properties if desired,
        # but for flexibility we'll just ask for n_samples unless user specifies otherwise.
        # Actually, for standard usage, we just call random(n).
        u_samples = self.engine.random(n=n_samples)
        
        configs = []
        
        for i in range(n_samples):
            row = u_samples[i]
            config = {}
            col_idx = 0
            
            # --- 1. Testbench / Context Parameters ---
            
            # fet_num (Discrete)
            # Map u -> Index
            u_fet = row[col_idx]; col_idx += 1
            fet_idx = int(u_fet * len(self.tech_nodes))
            # Clamp to safe bounds just in case u=1.0
            fet_idx = min(fet_idx, len(self.tech_nodes) - 1)
            fet_num = self.tech_nodes[fet_idx]
            config['fet_num'] = fet_num
            
            # Retrieve node constants
            t_const = TECH_CONSTANTS[fet_num]
            l_min = t_const['lmin']
            vdd_nom = t_const['vdd_nom']
            
            # VDD (Continuous)
            # LB: 0.9 * vdd_nom, UB: 1.1 * vdd_nom
            u_vdd = row[col_idx]; col_idx += 1
            vdd_lb = 0.9 * vdd_nom
            vdd_ub = 1.1 * vdd_nom
            vdd = vdd_lb + u_vdd * (vdd_ub - vdd_lb)
            config['vdd'] = vdd
            
            # VCM (Continuous)
            # LB: 0.15, UB: VDD - 0.15
            u_vcm = row[col_idx]; col_idx += 1
            vcm_lb = 0.15
            vcm_ub = vdd - 0.15
            # Ensure UB > LB (if VDD < 0.3, this breaks, but VDD min is ~0.63 for 7nm)
            if vcm_ub < vcm_lb: vcm_ub = vcm_lb
            vcm = vcm_lb + u_vcm * (vcm_ub - vcm_lb)
            config['vcm'] = vcm
            
            # Tempc (Continuous)
            # LB: -40, UB: 125
            u_temp = row[col_idx]; col_idx += 1
            temp_lb = globalsy.testbench_params['Tempc']['lb']
            temp_ub = globalsy.testbench_params['Tempc']['ub']
            tempc = temp_lb + u_temp * (temp_ub - temp_lb)
            config['tempc'] = tempc
            
            # --- 2. Environment Parameters (Log Sampling) ---
            
            # Helper for log sampling
            def log_sample(u, lb, ub):
                log_lb = np.log10(lb)
                log_ub = np.log10(ub)
                val_log = log_lb + u * (log_ub - log_lb)
                return 10 ** val_log

            # Rfeedback
            u_rf = row[col_idx]; col_idx += 1
            config['rfeedback_val'] = log_sample(
                u_rf, 
                globalsy.env_params['Rfeedback_val']['lb'],
                globalsy.env_params['Rfeedback_val']['ub']
            )

            # Rsrc
            u_rs = row[col_idx]; col_idx += 1
            config['rsrc_val'] = log_sample(
               u_rs,
               globalsy.env_params['R_src']['lb'],
               globalsy.env_params['R_src']['ub']
            )

            # Cload
            u_cl = row[col_idx]; col_idx += 1
            config['cload_val'] = log_sample(
               u_cl,
               globalsy.env_params['Cload_val']['lb'],
               globalsy.env_params['Cload_val']['ub']
            )
            
            # --- 3. Circuit Sizing Parameters ---
            
            for param in self.sizing_params:
                u_p = row[col_idx]
                col_idx += 1
                                
                if param.startswith('nA'): # Length, dependent on Lmin
                    # LB: 1.1 * Lmin, UB: 10 * Lmin
                    p_lb = 1.1 * l_min
                    p_ub = 10.0 * l_min
                    val = p_lb + u_p * (p_ub - p_lb)
                    config[param] = val
                    
                elif param.startswith('nB'): # Fin count / Width
                    # LB: 1, UB: 256
                    p_lb = 1
                    p_ub = 256
                    # Use floor to integer logic mapping [0,1] -> [1, 256]
                    # Map u to [1, 257) then floor? Or standard range
                    val = int(p_lb + u_p * (p_ub - p_lb + 0.999))
                    if val > 256: val = 256
                    config[param] = val
                    
                elif "bias" in param: # Vbias (e.g. vbiasn0, vbiasp0)
                    # LB: 0, UB: VDD
                    # Note: VDD varies per sample, so this is coupled!
                    p_lb = 0.0
                    p_ub = vdd
                    val = p_lb + u_p * (p_ub - p_lb)
                    config[param] = val

                elif param.startswith('nC'): # Internal C
                    # Logarithmic
                    p_lb = 100e-15
                    p_ub = 5e-12
                    config[param] = log_sample(u_p, p_lb, p_ub)

                elif param.startswith('nR'): # Internal R
                    # Logarithmic
                    p_lb = 500
                    p_ub = 500e3
                    config[param] = log_sample(u_p, p_lb, p_ub)
                    
                else: 
                     # Fallback for anything else (linear 0-1 or debug)
                     config[param] = u_p

            configs.append(config)
            
        return configs


if __name__ == "__main__":
    # Test stub
    test_params = ['nA1', 'nB1', 'vbiasn0', 'nR1']
    gen = SobolSizingGenerator(test_params)
    samples = gen.generate(5)
    
    import json
    print(json.dumps(samples, indent=2))
