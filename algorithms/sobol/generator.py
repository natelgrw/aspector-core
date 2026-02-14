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
        self.dim_sizing = len(self.sizing_params) # Aux dimension for TuRBO optimization
        
        # Initialize Scipy Sobol engine
        # scramble=True is generally recommended for better uniformity in lower sample counts
        self.engine = qmc.Sobol(d=self.dim, scramble=True, seed=seed)

    def generate(self, n_samples=None, u_samples=None, robust_env=False):
        """
        Generate n_samples of valid design configurations.

        Args:
            n_samples (int): Number of samples to generate.
            u_samples (np.array, optional): Pre-generated [0,1] samples of shape (n, dim).
                                            If dim == self.dim, full control.
                                            If dim == self.dim_sizing, fixes environment to Nominal OR Random if robust_env=True.
            robust_env (bool): If True and u_samples provides sizing only, the environment 
                               parameters are randomly sampled rather than fixed.

        Returns:
            list[dict]: List of dictionaries, each containing a full set of parameters.
        """
        
        mode = "full" 

        # Draw samples from the Unit Hypercube [0, 1]^d
        if u_samples is not None:
             n_samples = len(u_samples)
             input_dim = len(u_samples[0]) if isinstance(u_samples, list) else u_samples.shape[1]
             
             if input_dim == self.dim:
                 mode = "full"
             elif input_dim == self.dim_sizing:
                 if robust_env:
                     mode = "robust" # Sizing from input, Env from random/Sobol
                 else:
                     mode = "sizing_only" # Sizing from input, Env nominal
             else:
                  raise ValueError(f"Provided samples dim {input_dim} matches neither full ({self.dim}) nor sizing ({self.dim_sizing})")
        else:
             if n_samples is None: raise ValueError("Must provide n_samples if u_samples is None")
             u_samples = self.engine.random(n=n_samples)
        
        configs = []
        
        for i in range(n_samples):
            row = u_samples[i]
            config = {}
            col_idx = 0
            
            # Helper for log sampling
            def log_sample(u, lb, ub):
                log_lb = np.log10(lb)
                log_ub = np.log10(ub)
                val_log = log_lb + u * (log_ub - log_lb)
                return 10 ** val_log
            
            # --- 1. Testbench / Context Parameters ---
            
            if mode == "full":
                # fet_num (Discrete)
                u_fet = row[col_idx]; col_idx += 1
                fet_idx = int(u_fet * len(self.tech_nodes))
                fet_idx = min(fet_idx, len(self.tech_nodes) - 1)
                fet_num = self.tech_nodes[fet_idx]
                config['fet_num'] = fet_num
                
                # Retrieve node constants
                t_const = TECH_CONSTANTS[fet_num]
                vdd_nom = t_const['vdd_nom']
                
                # VDD
                u_vdd = row[col_idx]; col_idx += 1
                vdd_lb = 0.9 * vdd_nom
                vdd_ub = 1.1 * vdd_nom
                vdd = vdd_lb + u_vdd * (vdd_ub - vdd_lb)
                config['vdd'] = vdd
                
                # VCM
                u_vcm = row[col_idx]; col_idx += 1
                vcm_lb = 0.15
                vcm_ub = vdd - 0.15
                if vcm_ub < vcm_lb: vcm_ub = vcm_lb
                vcm = vcm_lb + u_vcm * (vcm_ub - vcm_lb)
                config['vcm'] = vcm
                
                # Tempc
                u_temp = row[col_idx]; col_idx += 1
                temp_lb = globalsy.testbench_params['Tempc']['lb']
                temp_ub = globalsy.testbench_params['Tempc']['ub']
                tempc = temp_lb + u_temp * (temp_ub - temp_lb)
                config['tempc'] = tempc
                
                # Rfeedback
                u_rf = row[col_idx]; col_idx += 1
                config['rfeedback_val'] = log_sample(u_rf, globalsy.env_params['Rfeedback_val']['lb'], globalsy.env_params['Rfeedback_val']['ub'])

                # Rsrc
                u_rs = row[col_idx]; col_idx += 1
                config['rsrc_val'] = log_sample(u_rs, globalsy.env_params['Rsrc_val']['lb'], globalsy.env_params['Rsrc_val']['ub'])
                
                # Cload
                u_cl = row[col_idx]; col_idx += 1
                config['cload_val'] = log_sample(u_cl, globalsy.env_params['Cload_val']['lb'], globalsy.env_params['Cload_val']['ub'])
                
            else: # mode == "sizing_only" or "robust"
                # Use Nominal/Fixed Values FIRST
                fet_num = self.tech_nodes[0] 
                config['fet_num'] = fet_num
                
                t_const = TECH_CONSTANTS[fet_num]
                vdd_nom = t_const['vdd_nom']
                
                config['vdd'] = vdd_nom
                config['vcm'] = vdd_nom / 2.0
                config['tempc'] = 27.0

                # Use geometric mean of bounds for log-scale parameters
                def get_geo_mean(param_key):
                    lb = globalsy.env_params[param_key]['lb']
                    ub = globalsy.env_params[param_key]['ub']
                    return math.sqrt(lb * ub)
                
                config['rfeedback_val'] = get_geo_mean('Rfeedback_val') 
                config['rsrc_val'] = get_geo_mean('Rsrc_val')
                config['cload_val'] = get_geo_mean('Cload_val')

                if mode == "robust":
                    # Overwrite with Random/Sampled Values
                    # We need randomness here. Since u_samples is fixed by TuRBO for sizing,
                    # we must generate new random numbers for env.
                    # We can use numpy random.
                    
                    rng = np.random.default_rng()
                    
                    # Random VDD (+/- 10%)
                    vdd_var = vdd_nom * 0.1
                    config['vdd'] = vdd_nom + rng.uniform(-1, 1) * vdd_var
                    
                    # Random Temp (-40 to 125)
                    config['tempc'] = rng.uniform(globalsy.testbench_params['Tempc']['lb'], 
                                                  globalsy.testbench_params['Tempc']['ub'])
                    
                    # Random VCM
                    vcm_lb = 0.15
                    vcm_ub = config['vdd'] - 0.15
                    if vcm_ub > vcm_lb:
                        config['vcm'] = rng.uniform(vcm_lb, vcm_ub)
                        
                    # Random Loads (Log)
                    def rand_log(key):
                        lb = globalsy.env_params[key]['lb']
                        ub = globalsy.env_params[key]['ub']
                        return 10 ** rng.uniform(np.log10(lb), np.log10(ub))
                        
                    config['rfeedback_val'] = rand_log('Rfeedback_val')
                    config['rsrc_val'] = rand_log('Rsrc_val')
                    config['cload_val'] = rand_log('Cload_val')
            
            # --- 2. Sizing Configs (Common) ---
            # Remove redundant block that was pasted here in previous edits?
            # The previous file read showed lines 180-210 contained duplicates of env params
            # We must be careful not to double-sample.
            
            # --- 3. Circuit Sizing Parameters ---
            
            for param in self.sizing_params:
                u_p = row[col_idx]
                col_idx += 1
                                
                if param.startswith('nA'): # Length, dependent on Lmin
                    # LB: 1.1 * Lmin, UB: 10 * Lmin
                    # t_const/l_min is retrieved above in either block
                    p_lb = 1.1 * t_const['lmin']
                    p_ub = 10.0 * t_const['lmin']
                    val = p_lb + u_p * (p_ub - p_lb)
                    config[param] = val
                    
                elif param.startswith('nB'): # Fin count / Width
                    # LB: 1, UB: 256
                    p_lb = 1
                    p_ub = 256
                    val = int(p_lb + u_p * (p_ub - p_lb + 0.999))
                    if val > 256: val = 256
                    config[param] = val
                    
                elif "bias" in param: # Vbias (e.g. vbiasn0, vbiasp0)
                    # LB: 0, UB: VDD (from config, set above)
                    p_lb = 0.0
                    p_ub = config['vdd']
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
            
            # Additional cleanup for 'mode' logic not in sample loop
            # If default mode, we had local vars. If full, we had local vars. All good.

            configs.append(config)
            
        return configs

    def inverse_map(self, df):
        """
        Maps physical parameters (from dataframe) back to Unit Hypercube [0,1]^d.
        Used for initializing TuRBO with existing data ("Sight" mode).
        
        Args:
            df (pd.DataFrame): Dataframe containing 'in_{param}' columns.
            
        Returns:
            torch.Tensor: Normalized X samples [N, dim_sizing]
        """
        import torch
        
        # Assume technology node is the primary/default one for sizing tasks
        fet_num = self.tech_nodes[0]
        t_const = TECH_CONSTANTS[fet_num]
        
        # Pre-calculate bounds (matching generate logic)
        # We need to know VDD for bias limits. 
        # In sizing_only mode, generator uses vdd_nom.
        current_vdd = t_const['vdd_nom'] 
        
        param_list = self.sizing_params
        X_rows = []
        
        for idx, row in df.iterrows():
            u_row = []
            
            # Helper for log inverse
            def log_inverse(val, lb, ub):
                log_lb = np.log10(lb)
                log_ub = np.log10(ub)
                u = (np.log10(val) - log_lb) / (log_ub - log_lb)
                return u

            try:
                for param in param_list:
                    # Handle column naming (collector uses 'in_' prefix)
                    col_name = f"in_{param}"
                    if col_name not in row:
                        # Fallback try without prefix
                        col_name = param
                    
                    if col_name not in row:
                        # Could be fixed parameter (fet_num context etc), skip or raise
                        continue
                        
                    val = row[col_name]
                    u = 0.5 # Default safety
                    
                    if param.startswith('nA'):
                        p_lb = 1.1 * t_const['lmin']
                        p_ub = 10.0 * t_const['lmin']
                        u = (val - p_lb) / (p_ub - p_lb)
                        
                    elif param.startswith('nB'):
                        p_lb = 1
                        p_ub = 256
                        # Mapping was: int(p_lb + u_p * (p_ub - p_lb + 0.999))
                        # Approx inverse: (val - p_lb) / (p_ub - p_lb)
                        u = (val - p_lb) / (p_ub - p_lb)
                        
                    elif "bias" in param:
                        p_lb = 0.0
                        p_ub = current_vdd
                        u = (val - p_lb) / (p_ub - p_lb)
                        
                    elif param.startswith('nC'):
                        p_lb = 100e-15
                        p_ub = 5e-12
                        u = log_inverse(val, p_lb, p_ub)
                        
                    elif param.startswith('nR'):
                        p_lb = 500
                        p_ub = 500e3
                        u = log_inverse(val, p_lb, p_ub)
                    
                    else:
                        u = val # Assuming already linear/normalized
                        
                    # Clamp
                    u = max(0.0, min(1.0, u))
                    u_row.append(u)
                    
                if len(u_row) == len(param_list):
                    X_rows.append(u_row)
                
            except Exception as e:
                # Skip bad rows
                continue
                
        return torch.tensor(X_rows, dtype=torch.double)



if __name__ == "__main__":
    # Test stub
    test_params = ['nA1', 'nB1', 'vbiasn0', 'nR1']
    gen = SobolSizingGenerator(test_params)
    samples = gen.generate(5)
    
    import json
    print(json.dumps(samples, indent=2))
