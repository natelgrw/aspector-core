"""
random_sample_turbo.py

Authors: dkochar, natelgrw, hhussein
Last Edited: 01/16/2026

TURBO-based circuit optimization for analog op-amp design. Integrates Bayesian
optimization with Spectre circuit simulation for automated parameter tuning.
Supports both single-ended and differential amplifier topologies with multi-output
logging for convergence tracking across optimization stages.
"""

import numpy as np
import os
import json
from collections import OrderedDict
from optimization.turbo_1 import Turbo1
from optimization.turbo_m import TurboM
from simulator.eval_engines.utils.netlist_to_graph import parse_netlist_to_graph, extract_sizing_map
import numpy as np
from simulator.eval_engines.spectre.measurements.single_ended_meas_man import *
from simulator.eval_engines.spectre.configs.config_env import *
from simulator import globalsy
import re


# ===== Interactive Configuration ===== #


# mapping of netlist selection choices to filenames
netlist_choices = {
    "1": "single_ended1",
    "2": "single_ended2",
    "3": "differential1",
    "4": "differential2"
}

# user input
netlist_choice = input("Select a netlist to optimize (1-4): \n 1: Single Ended Cascode Current Mirror \n 2: Single Ended Low Voltage Cascode Current Mirror \n 3: Differential PMOS Cascode Current Mirror \n 4: Differential Cascode \n")

while netlist_choice not in netlist_choices:
    netlist_choice = input("Invalid choice. Please select 1-4: \n 1: Single Ended Cascode Current Mirror \n 2: Single Ended Low Voltage Cascode Current Mirror \n 3: Differential PMOS Cascode Current Mirror \n 4: Differential Cascode \n")

netlist_name_base = netlist_choices[netlist_choice]
SCS_FILE_PATH = f"/homes/natelgrw/Documents/titan_foundation_model/demo_netlists/{netlist_name_base}.scs"
RESULTS_DIR = f"/homes/natelgrw/Documents/titan_foundation_model/results/{netlist_name_base}"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    os.makedirs(os.path.join(RESULTS_DIR, "graph"))
    os.makedirs(os.path.join(RESULTS_DIR, "sizings"))
    os.makedirs(os.path.join(RESULTS_DIR, "specs"))
    os.makedirs(os.path.join(RESULTS_DIR, "op_points"))
    
# Generate Graph
parse_netlist_to_graph(SCS_FILE_PATH, os.path.join(RESULTS_DIR, "graph"), topology_id=1)
SIZE_MAP = extract_sizing_map(SCS_FILE_PATH)

np.random.seed(2000)

fet_num = int(input("Input transistor card size in nm (e.g. 7): "))

vdd = float(input("Input supply voltage for simulation in volts (e.g. 1.0): "))
vcm = float(input("Input common mode voltage for simulation in volts (e.g. 0.5): "))
tempc = float(input("Input temperature for simulation in degrees Celsius (e.g. 27.0): "))

# transistor power states
region_mapping = {
    0: "cut-off",
    1: "triode",
    2: "saturation",
    3: "sub-threshold",
    4: "breakdown"
}

# netlist target specifications
specs_dict = {
    "gain": 1.0e5,
    "UGBW": 1.0e9,
    "PM": 60.0,
    "power": 1.0e-6,
}

# parameter bounds
shared_ranges = {
    'nA': (10e-9, 30e-9),
    'nB': (1, 20),
    'vbiasp': (0, 0.80),
    'vbiasn': (0, 0.80),
    'rr': (5e3, 1e7),
    'cc': (0.1e-12, 2.5e-12)
}

# reformatted specs for optimizer input
specs_id = list(specs_dict.keys())
specs_ideal = list(specs_dict.values())


# ===== Utility Functions ===== #


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
        Either \"differential\" or \"single_ended\" based on presence of Voutn.
    """
    with open(file_path, "r") as file:
        for line in file:
            if "Voutn" in line:
                return "differential"
        else:
            return "single_ended"


# ===== Optimization Function Class ===== #


class Levy:
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

    def __init__(self, dim, opt_params, params_id, specs_id, specs_ideal, vcm, vdd, tempc, ub, lb, yaml_path, fet_num, results_dir, netlist_name, size_map):
        
        self.dim = dim
        self.opt_params = opt_params        # Parameters being optimized (9)
        self.params_id = params_id          # All parameters for netlist (13)
        self.specs_id = specs_id
        self.specs_ideal = specs_ideal
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
        goal_spec = [float(e) for e in goal_spec]
        spec = [float(e) for e in spec]
        spec = np.array(spec)
        goal_spec = np.array(goal_spec)

        # normalized deviation calculation
        norm_spec = (spec-goal_spec)/(np.abs(goal_spec)+np.abs(spec))

        return norm_spec

    def reward(self, spec, goal_spec, specs_id):
        """
        Calculate the penalty-based reward (cost) from specifications.

        Converts specification deviations to a scalar reward score suitable for
        minimization. Penalties are weighted by specification importance:
        - Gain: 50x (critical for amplification)
        - UGBW: 30x (critical for bandwidth)
        - PM: 30x (critical for stability)
        - CMRR: 10x (important for rejection)
        - Power: 1x (secondary objective)

        Parameters:
        -----------
        spec : list or numpy array
            Measured specification values.
        goal_spec : list or numpy array
            Target specification values.
        specs_id : list
            Specification identifiers (names).
        
        Returns:
        --------
        float
            Total penalty score (weighted deviations). Lower is better.
        """
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0
        for i, rel_spec in enumerate(rel_specs):
            if specs_id[i] == "power" and rel_spec > 0:
                reward += np.abs(rel_spec)
            elif specs_id[i] == "gain" and rel_spec < 0:
                reward += 50.0 * np.abs(rel_spec)
            elif specs_id[i] == "UGBW" and rel_spec < 0:
                reward += 10.0 * np.abs(rel_spec)
            elif specs_id[i] == "PM" and rel_spec < 0:
                reward += 10.0 * np.abs(rel_spec)

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
    
        param_val = [OrderedDict(full_params)]

        # calls evaluate() to obtain simulation specs and sort them
        sim_env.ver_specs['results_dir'] = self.results_dir # Inject dir for meas man
        eval_result = sim_env.evaluate(param_val)
        
        # Error handling: check if evaluation returned valid results
        if not eval_result or len(eval_result) == 0:
            print(f"ERROR: Simulation returned empty list")
            globalsy.counterrrr += 1
            return float('inf')  # Return large penalty for failed evaluation
        
        cur_specs = OrderedDict(sorted(eval_result[0][1].items(), key=lambda k:k[0]))
        
        # Filter out dictionary outputs (op points, regions) from scalar specs used for optimization reward
        scalar_specs = OrderedDict()
        for k, v in cur_specs.items():
            if not isinstance(v, (dict, list)) and not k.startswith('z') and not k == 'valid':
                scalar_specs[k] = v
                
        # The legacy code relied on negative indexing: [:-5], [-5:-4], etc.
        # This was extremely fragile.
        # Original keys presumed: gain, funity, pm, power ... valid, zregion..., zzgm..., zzids..., zzvds..., zzvgs...
        #
        # Now that op points (zz...) are DICTIONARIES, we must not let them into the numpy array used for 'reward' or 'cur_specs'
        
        # Let's reconstruct 'cur_specs' to ONLY contain the optimization targets.
        # The 'specs_id' list tells us what specs we care about for reward.
        # specs_id = ['gain', 'UGBW', 'PM', 'power'] (based on specs_dict keys)
        
        # However, the measurement manager returns keys: 'gain', 'funity', 'pm', 'power'
        # 'funity' maps to 'UGBW' in spec list?
        # Let's look at legacy: 
        # dict1 = keys[:-5] -> The Main Specs
        # dict3 = keys[-5:-4] -> zregion ?
        # dict2 = keys[-4:] -> The other zz variables?
        
        # We need to construct the specific array expected by self.reward()
        # self.reward expects 'cur_specs' to match 'specs_id' order?
        # self.reward(spec, goal_spec, ...)
        # lookup(spec, goal_spec) -> subtracts arrays.
        
        # The variables 'dict1', 'dict2', 'dict3', 'dummy' seem to be legacy artifacts for data saving that we REPLACED with our JSON logging.
        # But 'cur_specs' is still used for calculating reward.
        
        # Let's EXTRACT just the scalar values needed for reward.
        # Note: In random_sample_turbo, specs_dict uses "UGBW", "PM".
        # In meas_man, keys are "funity", "pm".
        # This mapping seems implicit or broken in the original code unless sorted keys aligned perfectly.
        
        # Let's try to be robust.
        # Map known meas_man keys to specs_id keys if needed.
        # Or just rely on the fact that we have the raw dictionaries.
        
        # Map for single ended
        # valid=True/False is usually last or near op points.
        
        # Construct simplified vector for reward calculation
        # Order matters! specs_ideal order: gain, UGBW, PM, power
        reward_vals = []
        # Mapping: 
        # gain -> gain
        # UGBW -> funity
        # PM -> pm
        # power -> power
        
        val_gain = cur_specs.get('gain', 0)
        val_ugbw = cur_specs.get('funity', 0)
        val_pm   = cur_specs.get('pm', 0)
        val_pwr  = cur_specs.get('power', 0) # usually negative in results
        
        # Create array for reward function: [gain, ugbw, pm, power]
        # But wait, the original code did:
        # cur_specs = np.array(list(dict1.values()))[:-1]
        # dummy = cur_specs[0]
        # cur_specs[0] = cur_specs[1] ... swap?
        
        # This suggests the sorted order was [cmrr, funity, gain, integrated_noise, linearity, output_swing, pm, power, settle, slew, valid, vos, z...]
        # Sorted alphabetical!
        # cmrr, funity, gain ...
        # dict1 was [:-5] -> all scalars except last 5
        # dict1 values -> [cmrr, funity, gain, int_noise, lin, out_swing, pm, power, settle, slew, valid, vos]
        # [:-1] -> Removes last one (vos) or valid? if valid is boolean.
        
        # THIS IS A MESS. The original code was relying on alphabetical sorting of keys!
        # And we just added headers or changed types which broke existing flow.
        
        # FIX:
        # We only need 'reward1'.
        # self.reward uses: gain, UGBW, PM, power (specs_id)
        # We should just construct the array explicitly.
        
        reward_input = [val_gain, val_ugbw, val_pm, val_pwr]
        reward1 = self.reward(reward_input, self.specs_ideal, self.specs_id)

        # Log Sizing JSON
        sizing_dict = {}
        for pname, pval in full_params.items():
             # Basic grouping - assumes standard naming conventions nA/nB for transistors
             # and r.. c.. keys. This might need refinement for general usage.
             # but random_sample_turbo hardcodes ranges etc anyway for now.
             
             # Heuristic mapping for JSON structure
             # "sizing": { "M1": { ... } }
             pass
             
        # Actually building specific structure requested:
        # "sizing": { "M1": { "l": 14e-9, "nfin": 6 }, ... }
        # Need to know mapping from nA1/nB1 to M?
        # Typically M3/M0/M4/M6/M5/M1/M2 in single ended 1 scs.
        # Check Netlist again:
        # MM3 ... l=nA1 nfin=nB1
        # MM0 ... l=nA1 nfin=nB1
        # MM4 ... l=nA2 nfin=nB2
        # MM6 ... l=nA3 nfin=nB3
        # MM5 ... l=nA3 nfin=nB3
        # MM1 ... l=nA4 nfin=nB4
        # MM2 ... l=nA4 nfin=nB4
        
        # We can just dump flat params first or try to group. 
        # Requirement: "sizing_1.json", "sizing": { "M1": ... }
        # Let's dump the flat params into a structure that is helpful.
        
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

        globalsy.counterrrr += 1

        return reward1


# ===== TURBO-1 Optimization Setup and Execution ===== #


params_id = extract_parameter_names(SCS_FILE_PATH)

full_lb, full_ub, opt_lb, opt_ub, opt_params = build_bounds(params_id, shared_ranges, vdd, vcm, tempc, fet_num)

opamp_type = classify_opamp_type(SCS_FILE_PATH)
config_env = EnvironmentConfig(SCS_FILE_PATH, opamp_type, specs_dict, params_id, full_lb, full_ub)
yaml_path = config_env.write_yaml_configs()

f = Levy(len(opt_lb), opt_params, params_id, specs_id, specs_ideal, vcm, vdd, tempc, opt_ub, opt_lb, yaml_path, fet_num, RESULTS_DIR, netlist_name_base, SIZE_MAP)

turbo1 = Turbo1(
    f=f,                           # Objective function handle
    lb=opt_lb,                     # Lower bounds array (optimized params only)
    ub=opt_ub,                     # Upper bounds array (optimized params only)
    n_init=20,                     # Initial design points from Latin hypercube
    max_evals=2000,                # Maximum total evaluations allowed
    batch_size=5,                  # Points evaluated per batch
    verbose=True,                  # Print batch-level progress
    use_ard=True,                  # Automatic Relevance Determination for GP
    max_cholesky_size=2000,        # Cholesky vs Lanczos threshold
    n_training_steps=30,           # ADAM optimization steps per batch
    min_cuda=10.40,                # CUDA memory threshold
    device="cpu",                  # Compute device
    dtype="float32",               # Floating point precision
)

turbo1.optimize()

X = turbo1.X                       
fX = turbo1.fX                     
ind_best = np.argmin(fX)           
f_best, x_best = fX[ind_best], X[ind_best, :]

print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))

config_env.del_yaml()


# ===== TURBO-M Multi-Region Optimization ===== #


# Alternative: TURBO-M for multi-region optimization
# Uncomment to use TURBO-M with multiple parallel trust regions instead of TURBO-1
# turbo_m = TurboM(
#     f=f,                           # Objective function handle
#     lb=lb,                         # Lower bounds array
#     ub=ub,                         # Upper bounds array
#     n_init=10,                     # Initial design points per trust region
#     max_evals=300,                 # Maximum total evaluations
#     n_trust_regions=5,             # Number of parallel trust regions
#     batch_size=10,                 # Points evaluated per batch
#     verbose=True,                  # Print batch-level progress
#     use_ard=True,                  # Automatic Relevance Determination for GP
#     max_cholesky_size=200,         # Cholesky vs Lanczos threshold
#     n_training_steps=50,           # ADAM optimization steps per batch
#     min_cuda=10.40,                # CUDA memory threshold
#     device="cpu",                  # Compute device
#     dtype="float32",               # Floating point precision
# )
#
# turbo_m.optimize()
#
# X = turbo_m.X  # Evaluated points
# fX = turbo_m.fX  # Observed values
# ind_best = np.argmin(fX)
# f_best, x_best = fX[ind_best], X[ind_best, :]
#
# print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
