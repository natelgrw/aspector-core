"""
extractor.py

Author: natelgrw
Last Edited: 01/24/2026

Extractor engine for processing simulation results and calculating rewards.
Handles mapping of parameters, specification verification, and JSON logging.
"""

import numpy as np
import os
import json
import re
from collections import OrderedDict
from simulator import globalsy
from simulator.eval_engines.spectre.measurements.single_ended_meas_man import OpampMeasMan

def extract_parameter_names(scs_file):
    """
    Extracts all parameter names from a Spectre netlist file.

    Parameters:
    -----------
    scs_file : str
        Path to the Spectre netlist (.scs) file.
    
    Returns:
    --------
    list
        List of parameter names found in the netlist parameters declaration.
    """
    with open(scs_file, "r") as file:
        for line in file:
            if line.strip().startswith("parameters"):
                matches = re.findall(r'(\w+)=', line)
                matches.remove("dc_offset")
                matches.remove("gain_n")
                matches.remove("use_tran")
                matches.remove("rfeedback_val")
                return matches
    return []

def build_bounds(params_id, shared_ranges, vdd, vcm, tempc, fet_num):
    """
    Constructs parameter bounds for optimization.

    Maps parameter names to bounds using predefined ranges and fixed values
    for temperature, supply voltage, and transistor size. Returns both full
    bounds (for config) and reduced bounds (for optimization).

    Parameters:
    -----------
    params_id : list
        List of parameter names to create bounds for.
    shared_ranges : dict
        Dictionary of parameter type ranges.
    vdd : float
        Supply voltage (fixed parameter).
    vcm : float
        Common mode voltage (fixed parameter).
    tempc : float
        Temperature in Celsius (fixed parameter).
    fet_num : int
        Transistor size in nm (fixed parameter).
    
    Returns:
    --------
    tuple of (np.array, np.array, np.array, np.array, list)
        (full_lb, full_ub, opt_lb, opt_ub, opt_params) where:
        - full_lb/full_ub: bounds for all parameters (for config_env)
        - opt_lb/opt_ub: bounds for optimized parameters only (for TURBO)
        - opt_params: names of parameters being optimized
    
    Raises:
    -------
    ValueError
        If a parameter name is not recognized.
    """
    full_lb, full_ub = [], []
    opt_lb, opt_ub, opt_params = [], [], []
    
    for pname in params_id:
        if pname.startswith("nA"):
            low, high = shared_ranges['nA']
        elif pname.startswith("nB"):
            low, high = shared_ranges['nB']
        elif "biasp" in pname:
            low, high = shared_ranges['vbiasp']
        elif "biasn" in pname:
            low, high = shared_ranges['vbiasn']
        elif pname.startswith("nC"):
            low, high = shared_ranges['cc']
        elif pname.startswith("nR"):
            low, high = shared_ranges['rr']
        elif pname.startswith("vdd"):
            low, high = (vdd, vdd)
        elif pname.startswith("vcm"):
            low, high = (vcm, vcm)
        elif pname.startswith("tempc"):
            low, high = (tempc, tempc)
        elif pname.startswith("fet_num"):
            low, high = (fet_num, fet_num)
        else:
            raise ValueError(f"Parameter {pname} not recognized in shared_ranges.")
        
        # Add to full bounds (for config_env)
        full_lb.append(low)
        full_ub.append(high)
        
        # Only add to optimization bounds if not fixed
        if low < high:
            opt_lb.append(low)
            opt_ub.append(high)
            opt_params.append(pname)

    return np.array(full_lb), np.array(full_ub), np.array(opt_lb), np.array(opt_ub), opt_params

def classify_opamp_type(file_path):
    """
    Classifies op-amp topology based on netlist structure.

    Parameters:
    -----------
    file_path : str
        Path to the Spectre netlist file.
    
    Returns:
    --------
    str
        Either "differential" or "single_ended" based on presence of Voutn.
    """
    with open(file_path, "r") as file:
        for line in file:
            if "Voutn" in line:
                return "differential"
        else:
            return "single_ended"


class Extractor:
    """
    Objective function wrapper for circuit optimization.

    Encapsulates netlist simulation and performance evaluation for use with
    TURBO Bayesian optimization. Converts raw simulation results to reward
    (penalty) scores for minimization.
    
    Initialization Parameters:
    --------------------------
    dim : int
        Dimension of parameter space.
    params_id : list
        Parameter names.
    specs_id : list
        Specification names.
    specs_ideal : list
        Target specification values.
    vcm : float
        Common mode voltage.
    vdd : float
        Supply voltage.
    tempc : float
        Temperature in Celsius.
    ub : numpy array
        Upper bounds for parameters.
    lb : numpy array
        Lower bounds for parameters.
    yaml_path : str
        Path to YAML configuration file.
    fet_num : int
        Transistor size in nm.
    """

    def __init__(self, dim, opt_params, params_id, specs_id, specs_ideal, specs_weights, sim_flags, vcm, vdd, tempc, ub, lb, yaml_path, fet_num, results_dir, netlist_name, size_map):
        
        self.dim = dim
        self.opt_params = opt_params        # Parameters being optimized (9)
        self.params_id = params_id          # All parameters for netlist (13)
        self.specs_id = specs_id
        self.specs_ideal = specs_ideal
        self.specs_weights = specs_weights
        self.sim_flags = sim_flags
        self.vcm = vcm
        self.vdd = vdd
        self.tempc = tempc
        self.ub = ub
        self.lb = lb
        self.yaml_path = yaml_path
        self.fet_num = fet_num
        self.results_dir = results_dir
        self.netlist_name = netlist_name
        self.size_map = size_map

    def lookup(self, spec, goal_spec):
        """
        Calculate normalized performance deviation from target specifications.

        Computes the deviation of measured specifications from ideal targets using
        normalized relative error: (spec - goal_spec) / (|goal_spec| + |spec|).
        This normalization prevents division by zero and scales deviations consistently.

        Parameters:
        -----------
        spec : list or numpy array
            Measured specification values.
        goal_spec : list or numpy array
            Target specification values.
        
        Returns:
        --------
        numpy array
            Normalized deviations for each specification. Positive = exceeds target,
            Negative = below target.
        """
        norm_specs = []
        for s, g in zip(spec, goal_spec):
            # Check for range measurement (tuple/list of length 2)
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) == 2:
                 # Check if target is also a range
                 if isinstance(g, (list, tuple, np.ndarray)) and len(g) == 2:
                     # Containment Optimization
                     val_min = (g[0] - s[0]) / (abs(g[0]) + abs(s[0]) + 1e-9)
                     val_max = (s[1] - g[1]) / (abs(g[1]) + abs(s[1]) + 1e-9)
                     norm_specs.append(min(val_min, val_max))
                 else:
                     # Width Optimization
                     width_s = abs(s[1] - s[0])
                     width_g = float(g)
                     val = (width_s - width_g) / (abs(width_g) + abs(width_s) + 1e-9)
                     norm_specs.append(val)
            else:
                 # Scalar Optimization
                 s_val = float(s) if s is not None else 0.0
                 g_val = float(g)
                 val = (s_val - g_val) / (abs(g_val) + abs(s_val) + 1e-9)
                 norm_specs.append(val)

        return np.array(norm_specs)

    def reward(self, spec, goal_spec, specs_id, specs_weights):
        """
        Calculate the penalty-based reward (cost) from specifications.
        """
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0
        
        # Define Optimization Direction (Maximize vs Minimize)
        # Minimize: Penalty if rel_spec > 0 (Measured > Target). 
        # Maximize: Penalty if rel_spec < 0 (Measured < Target).
        minimize_specs = ["power", "integrated_noise", "settling_time", "vos"]
        
        for i, rel_spec in enumerate(rel_specs):
            s_name = specs_id[i]
            s_weight = specs_weights[i]
            
            if s_name in minimize_specs:
                if rel_spec > 0: # Bad
                    reward += s_weight * np.abs(rel_spec)
            else: # Maximize (gain, ugbw, pm, cmrr, etc.)
                if rel_spec < 0: # Bad
                    reward += s_weight * np.abs(rel_spec)

        return reward

    def __call__(self, x):
        """
        Main evaluation function for TURBO optimizer.

        Converts parameter vector to circuit netlist, runs Spectre simulation,
        extracts performance specifications, calculates penalties, and logs
        evaluation metadata. Supports multi-stage logging based on evaluation count.

        Parameters:
        -----------
        x : numpy array
            Parameter vector of length self.dim. Discrete parameters (nB) are rounded.
        
        Returns:
        --------
        float
            Penalty score from reward() function. Lower is better.
        
        Notes:
        ------
        - nB (transistor multiplier) is rounded to nearest integer
        - Multi-stage logging: out1.txt (0-199), out11.txt (200-1199), out12.txt (1200-1999)
        - Increments globalsy.counterrrr for tracking total evaluations
        """
        assert len(x) == self.dim
        assert x.ndim == 1

        # creation of cadence simulation environment
        sim_env = OpampMeasMan(self.yaml_path)

        # Print iteration steps
        iter_num = globalsy.counterrrr + 1 # counter starts at 0 for filename, display 1-based
        print(f" Iteration {iter_num}:")
        print(f"   [+] Making measurements...", end="\r")

        sample = x.copy()

        # round discrete parameters (only for optimized params)
        for i, param in enumerate(self.opt_params):
            if param.startswith('nB'): 
                sample[i] = round(sample[i])

        # Create full parameter dictionary by mapping optimized params to all params
        full_params = {}
        opt_idx = 0
        for pname in self.params_id:
            if pname in self.opt_params:
                full_params[pname] = sample[opt_idx]
                opt_idx += 1
            elif pname.startswith("vdd"):
                full_params[pname] = self.vdd
            elif pname.startswith("vcm"):
                full_params[pname] = self.vcm
            elif pname.startswith("tempc"):
                full_params[pname] = self.tempc
            elif pname.startswith("fet_num"):
                full_params[pname] = self.fet_num
    
        # Inject Simulation Control Flags dependent on user selection
        # These are passed to the Jinja2 template to conditionally include analysis blocks
        full_params["run_ac"]    = 1 if self.sim_flags['ac'] else 0
        full_params["run_dc"]    = 1 if self.sim_flags['dc'] else 0
        full_params["run_noise"] = 1 if self.sim_flags['noise'] else 0
        full_params["run_tran"]  = 1 if self.sim_flags['tran'] else 0

        param_val = [OrderedDict(full_params)]

        # calls evaluate() to obtain simulation specs and sort them
        sim_env.ver_specs['results_dir'] = self.results_dir # Inject dir for meas man
        eval_result = sim_env.evaluate(param_val)
        
        print(f"   [+] Completed measurements      ")
        
        # Error handling: check if evaluation returned valid results
        if not eval_result or len(eval_result) == 0:
            print(f"ERROR: Simulation returned empty list")
            globalsy.counterrrr += 1
            return float('inf')  # Return large penalty for failed evaluation
        
        cur_specs = OrderedDict(sorted(eval_result[0][1].items(), key=lambda k:k[0]))
        
        # Construct Spec Vector matching the Optimizer's ID list
        reward_input = []
        for s_name in self.specs_id:
            val = cur_specs.get(s_name)
            if val is None:
                 # Logic for missing specs (e.g. if simulation didn't run effectively)
                 # Assign a terrible value to discourage this region or indicate failure
                 if s_name in ["power", "integrated_noise", "settling_time", "vos"]:
                     val = 1e9 # High value bad
                 else:
                     val = -1e9 # Low value bad
            reward_input.append(val)
        
        reward1 = self.reward(reward_input, self.specs_ideal, self.specs_id, self.specs_weights)

        # Build structured sizing from map
        structured_sizing = {}
        if self.size_map:
            for comp, params in self.size_map.items():
                comp_props = {}
                for prop, val_expr in params.items():
                    # Handle Jinja variables {{var}} or direct var
                    clean_var = val_expr.replace('{{', '').replace('}}', '').strip()
                    if clean_var in full_params:
                        comp_props[prop] = full_params[clean_var]
                    else:
                        # Try literal
                        try:
                            comp_props[prop] = float(clean_var)
                        except ValueError:
                            comp_props[prop] = clean_var
                structured_sizing[comp] = comp_props
        
        final_sizing = structured_sizing if structured_sizing else full_params

        # Build env and bias dictionaries
        env_dict = {
            "vdd": self.vdd,
            "vcm": self.vcm,
            "tempc": self.tempc,
            "fet_num": self.fet_num
        }
        
        bias_dict = {}
        
        # Add any other bias parameters found in full_params
        # Look for keys starting with vbias, ibias, or specifically vbiasn0 etc. if not already there
        for k, v in full_params.items():
            if k not in env_dict:
                if k.lower().startswith('vbias') or k.lower().startswith('ibias') or 'bias' in k.lower():
                     bias_dict[k] = v

        sizing_data = {
            "topology_id": 1,
            "netlist": self.netlist_name,
            "simulation_id": globalsy.counterrrr,
            "sizing": final_sizing,
            "bias": bias_dict,
            "env": env_dict
        }
        
        # Cleanup full_params vs bias - remove bias from sizing section if present
        for k in ["vdd", "vcm", "tempc", "fet_num"]:
            if k in sizing_data["sizing"]:
                del sizing_data["sizing"][k]
        
        # Write JSON
        sizing_file = os.path.join(self.results_dir, "sizings", f"sizing_{globalsy.counterrrr}.json")
        with open(sizing_file, 'w') as f:
            json.dump(sizing_data, f, indent=2)

        # Process Specs and Operating Points
        raw_specs = eval_result[0][1] if eval_result and len(eval_result) > 0 and len(eval_result[0]) > 1 else {}
        
        main_specs = {}
        op_points = {}
        
        # Helper to safely merge dictionary data
        def merge_op_data(target_dict, param_name, data_dict):
            # data_dict is expected to be { "MM0": 1.23, "MM1": 4.56 ... }
            if isinstance(data_dict, dict):
                for comp_name, val in data_dict.items():
                    if comp_name not in target_dict:
                        target_dict[comp_name] = {}
                    target_dict[comp_name][param_name] = val
            elif isinstance(data_dict, list):
                 # Fallback if meas_man.py wasn't updated or returns list
                 pass

        for k, v in raw_specs.items():
            if k.startswith("z"):
                # Clean up key name: remove 'zz' prefix and '_MM' suffix
                # keys are like zzgm_MM, zzids_MM
                # v is now a dictionary { "MM0": x, ... }
                
                clean_k = k.lstrip('z')
                if clean_k.endswith('_MM'):
                    clean_k = clean_k[:-3] # remove _MM
                    
                # We want op_points to be { "MM0": { "gm": ..., "ids": ... } }
                merge_op_data(op_points, clean_k, v)
            else:
                main_specs[k] = v

        # Log Specs JSON
        spec_data = {
            "topology_id": 1,
            "netlist": self.netlist_name,
            "simulation_id": globalsy.counterrrr,
            "specs": main_specs
        }
        
        spec_file = os.path.join(self.results_dir, "specs", f"spec_{globalsy.counterrrr}.json")
        with open(spec_file, 'w') as f:
            json.dump(spec_data, f, indent=2)

        # Log Operating Points JSON
        op_point_data = {
            "topology_id": 1,
            "netlist": self.netlist_name,
            "simulation_id": globalsy.counterrrr,
            "operating_points": op_points
        }
        
        op_point_file = os.path.join(self.results_dir, "op_points", f"op_point_{globalsy.counterrrr}.json")
        with open(op_point_file, 'w') as f:
            json.dump(op_point_data, f, indent=2)

        print(f"   [+] Exporting sizings, specs, op_points\n")

        globalsy.counterrrr += 1

        return reward1
