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
import uuid
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
    excluded_params = {
        "dc_offset", "use_tran", "use_sine", "mode_unity", 
        "run_gatekeeper", "run_full_char", "fet_num", "vdd", "vcm", "tempc",
        "sim_tier_1", "sim_tier_2"
    }

    with open(scs_file, "r") as file:
        for line in file:
            if line.strip().startswith("parameters"):
                matches = re.findall(r'(\w+)=', line)
                # Filter out excluded parameters (control flags and env vars)
                return [m for m in matches if m not in excluded_params and not m.startswith("sim_")]
    return []

def build_bounds(params_id, shared_ranges, vdd, vcm, tempc, fet_num):
    """
    Constructs parameter bounds for optimization using specific rules.
    """
    full_lb, full_ub = [], []
    opt_lb, opt_ub, opt_params = [], [], []
    
    for pname in params_id:
        
        # --- Basic Parameters ---
        if pname.startswith("nA"): # Length (L) - Bound handled by generator mostly, but here for safety
            # Using shared_ranges defaults which should be set to rough L min/max if not 
            # strictly enforced by generator. But ideally generator sets this per technology.
            low, high = shared_ranges['nA'] 
        elif pname.startswith("nB"): # Nfin
            low, high = (1, 256)
        elif "biasp" in pname or "biasn" in pname or pname.startswith("vbias"): # Vbias
            low, high = (0, vdd) # strictly 0 to VDD
        elif pname.startswith("nC"): # C_internal
            low, high = (100e-15, 5e-12) # 100fF to 5pF
        elif pname.startswith("nR"): # R_internal
            low, high = (500, 500e3) # 500 Ohm to 500 kOhm
            
        # --- Environment Parameters ---
        elif pname == "rfeedback_val":
            low, high = (1e3, 1e6) # 1k to 1M
        elif pname == "rsrc_val":
            low, high = (50, 100e3) # 50 to 100k
        elif pname == "cload_val":
            low, high = (10e-15, 10e-12) # 10fF to 10pF
            
        # --- Fixed/Environment Variables (pass-through) ---
        elif pname.startswith("vdd"):
            low, high = (vdd, vdd)
        elif pname.startswith("vcm"):
            low, high = (vcm, vcm)
        elif pname.startswith("tempc"):
            low, high = (tempc, tempc)
        elif pname.startswith("fet_num"):
            low, high = (fet_num, fet_num)
        else:
            print(f"Warning: Parameter {pname} using generic fallback bounds.")
            low, high = (-1e9, 1e9)
        
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
    Objective function wrapper for circuit simulation.

    Encapsulates netlist simulation and performance evaluation for use with
    Sobol sampling. Converts raw simulation results to reward (penalty) scores for logging.
    
    Initialization Parameters:
    --------------------------
    dim : int
        Dimension of parameter space.
    params_id : list
        Parameter names.
    specs_id : list
        Specification names (from optimization target list - often empty now).
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

    def __init__(self, dim, opt_params, params_id, specs_id, specs_ideal, specs_weights, sim_flags, vcm, vdd, tempc, ub, lb, yaml_path, fet_num, results_dir, netlist_name, size_map, rfeedback=1e7, rsrc=50, cload=1e-12):
        
        self.dim = dim
        self.opt_params = opt_params        # Parameters being varied
        self.params_id = params_id          # All parameters for netlist
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
        
        # Passive component defaults
        self.rfeedback = rfeedback
        self.rsrc = rsrc
        self.cload = cload

    def lookup(self, spec, goal_spec):
        """
        Calculate normalized performance deviation from target specifications.
        Kept for backward compatibility if we want to calculate rewards for analysis,
        even if not used for active optimization loop steering.
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
        if not specs_id:
            return 0.0
            
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0
        
        # Define Optimization Direction (Maximize vs Minimize)
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

    def __call__(self, x, sim_id=None):
        """
        Main execution function for Sobol sampling loop.

        Converts parameter vector to circuit netlist, runs Spectre simulation,
        logs evaluation metadata.

        Parameters:
        -----------
        x : numpy array or dict
            Parameter vector of length self.dim.
        sim_id : int, optional
            Explicit simulation ID. If None, uses globalsy.counterrrr.
        
        Returns:
        --------
        float
            Dummy reward score (0.0).
        """
        # Handle Dictionary Input (Direct from Generator) - PREFERRED PATH
        if isinstance(x, dict):
            full_params = x.copy()
            
            # Ensure discrete params are int
            for pname, val in full_params.items():
                if pname.startswith("nB"):
                    full_params[pname] = int(round(val))
                    
        else:
            # Legacy/Vector Input Path
            assert len(x) == self.dim
            assert x.ndim == 1

            sample = x.copy()
            # ... (rest of legacy mapping logic if ever needed) ...
            # For now, let's just implement the vector mapping assuming legacy usage
            
            for i, param in enumerate(self.opt_params):
                 if param.startswith('nB'): 
                    sample[i] = round(sample[i])

            full_params = {}
            opt_idx = 0
            
            for pname in self.params_id:
                if pname in self.opt_params:
                    full_params[pname] = sample[opt_idx]
                    opt_idx += 1
            
            # Helper for leftovers
            while opt_idx < len(self.opt_params):
                pname = self.opt_params[opt_idx]
                if pname not in full_params:
                     full_params[pname] = sample[opt_idx]
                opt_idx += 1

        # Common Logic: creation of cadence simulation environment
        sim_env = OpampMeasMan(self.yaml_path)

        # Print iteration steps
        if sim_id is not None:
             iter_num = sim_id
        else:
             iter_num = globalsy.counterrrr + 1 
        
        print(f" Iteration {iter_num}:")
        print(f"   [+] Making measurements...", end="\r")

        # Now Update self state from full_params if present (Sobol mode)
        # OR use current self state if not present (Fixed mode)
        if "fet_num" in full_params: self.fet_num = full_params["fet_num"]
        else: full_params["fet_num"] = self.fet_num
            
        if "vdd" in full_params: self.vdd = full_params["vdd"]
        else: full_params["vdd"] = self.vdd
            
        if "vcm" in full_params: self.vcm = full_params["vcm"]
        else: full_params["vcm"] = self.vcm
            
        if "tempc" in full_params: self.tempc = full_params["tempc"]
        else: full_params["tempc"] = self.tempc

        # Ensure passive defaults if not swept/provided by generator
        if "vcm" in full_params: self.vcm = full_params["vcm"]
        if "tempc" in full_params: self.tempc = full_params["tempc"]

        # Ensure passive defaults if not swept/provided by generator
        if "rfeedback_val" not in full_params: full_params["rfeedback_val"] = self.rfeedback
        if "rsrc_val" not in full_params: full_params["rsrc_val"] = self.rsrc
        if "cload_val" not in full_params: full_params["cload_val"] = self.cload
        
        # Inject Simulation Control Flags dependent on user selection
        # run_gatekeeper typically enables basic startup checks (DC, STB)
        full_params["run_gatekeeper"] = 1 # Always run primary gatekeeper
        
        # RUN FULL CHARACTERIZATION (Tier 2 enabled)
        full_params["run_full_char"] = 1
        
        param_val = [OrderedDict(full_params)]

        # calls evaluate() to obtain simulation specs and sort them
        sim_env.ver_specs['results_dir'] = self.results_dir # Inject dir for meas man
        eval_result = sim_env.evaluate(param_val)
        
        print(f"   [+] Completed measurements      ")
        
        # Error handling: check if evaluation returned valid results
        if not eval_result or len(eval_result) == 0:
            print(f"ERROR: Simulation returned empty list")
            globalsy.counterrrr += 1
            return 0.0
        
        cur_specs = OrderedDict(sorted(eval_result[0][1].items(), key=lambda k:k[0]))
        
        print("\n [DEBUG] Extracted Specs (Full Characterization):")
        for k, v in cur_specs.items():
            print(f"   - {k}: {v}")
        print("-" * 50)
        
        # Calculate Reward (Optional/Dummy if specs_id empty)
        reward_input = []
        if self.specs_id:
            for s_name in self.specs_id:
                val = cur_specs.get(s_name)
                if val is None:
                    if s_name in ["power", "integrated_noise", "settling_time", "vos"]:
                        val = 1e9 # High value bad
                    else:
                        val = -1e9 # Low value bad
                reward_input.append(val)
            
            reward1 = self.reward(reward_input, self.specs_ideal, self.specs_id, self.specs_weights)
        else:
            reward1 = 0.0

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
            "topology_id": self.netlist_name, # Updated to netlist name per user request/standardization
            "netlist": self.netlist_name,
            "sim_id": iter_num,
            "sizing": final_sizing,
            "bias": bias_dict,
            "env": env_dict
        }
        
        # Cleanup full_params vs bias - remove bias from sizing section if present
        for k in ["vdd", "vcm", "tempc", "fet_num"]:
            if k in sizing_data["sizing"]:
                del sizing_data["sizing"][k]

        # --- Custom Cleanup per User Request ---
        # 1. Remove unwanted components (Rshunt, Runity, Rsw) and individual Rsrc
        comps_to_remove = []
        for comp in sizing_data["sizing"]:
            if comp.startswith("Rshunt") or comp.startswith("Rsw") or comp.startswith("R_unity") or comp.startswith("Rsrc_"):
                comps_to_remove.append(comp)
        
        for comp in comps_to_remove:
            del sizing_data["sizing"][comp]

        # 2. Add/Consolidate desired components (Rfeedback, Cload, Rsrc)
        if "rfeedback_val" in full_params:
            sizing_data["sizing"]["Rfeedback"] = {"r": full_params["rfeedback_val"]}
        if "cload_val" in full_params:
             sizing_data["sizing"]["Cload"] = {"c": full_params["cload_val"]}
        if "rsrc_val" in full_params:
             sizing_data["sizing"]["Rsrc"] = {"r": full_params["rsrc_val"]}
        # ---------------------------------------
        
        # Generate Unique ID for this simulation instance
        sim_uuid = str(uuid.uuid4())

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
                    target_dict[comp_name][param_name] = float('%.6g' % val) if isinstance(val, (int, float)) else val
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
                # Standardize Specs to 6 sig figs
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    main_specs[k] = float('%.6g' % v)
                elif isinstance(v, tuple):
                    main_specs[k] = [float('%.6g' % x) if isinstance(x, (int, float)) else x for x in v]
                else:
                    main_specs[k] = v

        # Consolidate Simulation Result into Single Object
        simulation_result = {
            "id": sim_uuid,
            "topology_id": 1,
            "netlist": self.netlist_name,
            "parameters": sizing_data["sizing"], # Renamed from 'sizing' to 'parameters' per user intent
            "bias": sizing_data["bias"],
            "env": sizing_data["env"],
            "specs": main_specs,
            "operating_points": op_points
        }

        # Write to "simulations" directory
        sim_dir = os.path.join(self.results_dir, "simulations")
        os.makedirs(sim_dir, exist_ok=True)
        
        sim_file = os.path.join(sim_dir, f"{sim_uuid}.json")
        with open(sim_file, 'w') as f:
            json.dump(simulation_result, f, indent=2)

        print(f"   [+] Exporting consolidated simulation results (ID: {sim_uuid})\n")

        # globalsy.counterrrr += 1 # No longer needed/safe

        return reward1
