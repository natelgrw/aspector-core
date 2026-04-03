"""
extractor.py

Author: natelgrw
Last Edited: 04/01/2026

Extractor wrapper for simulation orchestration and result shaping.
"""

import numpy as np
import os
import json
import uuid
import re
import hashlib
import time
import errno
from collections import OrderedDict
import importlib

def extract_parameter_names(scs_file):
    """
    Extract sampled parameter names from a spectre netlist.

    Parameters:
    -----------
    scs_file : str
        Path to the spectre netlist (.scs) file.
    
    Returns:
    --------
    list
        Parameter names in the netlist parameters declaration.
    """
    non_optimizable_params = {
        "dc_offset", "is_hp", "n_state", "p_state",
        "run_gatekeeper", "run_full_char", "loop_mode", "fet_num", "vdd", "vcm", "tempc", "cload_val"
    }

    with open(scs_file, "r") as file:
        for line in file:
            if line.strip().startswith("parameters"):
                matches = re.findall(r'(\w+)=', line)
                # keep only optimizable design variables; runtime context is handled in __call__
                return [m for m in matches if m not in non_optimizable_params and not m.startswith("sim_")]
    return []

def classify_opamp_type(file_path):
    """
    Classify op-amp topology from the netlist text.

    Parameters:
    -----------
    file_path : str
        Path to the spectre netlist file.
    
    Returns:
    --------
    str
        "differential" if Voutn exists, else "single_ended".
    """
    with open(file_path, "r") as file:
        for line in file:
            if "Voutn" in line:
                return "differential"
        else:
            return "single_ended"


class Extractor:

    _meas_man_cache = {}
    _iter_counter = 0

    def __init__(self, dim, opt_params, params_id, specs_id, specs_ideal, specs_weights, config, results_dir, netlist_name, size_map, cload_val=1e-12, mode="mass_collection"):
        """
        Initializes the Extractor class which serves as a wrapper for running simulations and extracting specs 
        based on given parameters.

        Parameters:
        -----------
        dim (int): Dimensionality of the optimization problem (number of parameters).
        opt_params (list): List of parameter names corresponding to the optimization vector.
        params_id (list): List of all parameter names in the order they appear in the netlist.
        specs_id (list): List of spec names to extract and return.
        specs_ideal (list): List of ideal target values for the specs, used for reward calculation.
        specs_weights (list): List of weights for each spec in the reward calculation.
        config (dict): Configuration dictionary for the simulation environment.
        results_dir (str): Directory path where simulation results will be stored.
        netlist_name (str): Name of the netlist being simulated, used for identification.
        size_map (dict): Optional mapping of component names to their parameter expressions for structured sizing.
        cload_val (float): Default load capacitance value to use if not specified in parameters.
        mode (str): Operation mode, either "mass_collection" for large batch runs or "single_run" for individual simulations with per-sim JSON outputs.
        """
        self.mode = mode
        self.dim = dim
        self.opt_params = opt_params
        self.params_id = params_id
        self.specs_id = specs_id
        self.specs_ideal = specs_ideal
        self.specs_weights = specs_weights
        self.vcm = None
        self.vdd = None
        self.tempc = None
        self.fet_num = None
        self.is_hp = None
        self.n_state = None
        self.p_state = None
        self.config = config
        self.results_dir = results_dir
        self.netlist_name = netlist_name
        self.size_map = size_map

        self.cload_val = cload_val

    def _get_opamp_meas_man(self):
        """
        Get or cache the OpampMeasMan class to avoid 6.4M dynamic imports (Titan-Killer #2).
        
        Returns:
        --------
        class: OpampMeasMan measurement manager class.
        """
        tb_module_name = self.config['measurement']['testbenches']['ac_dc']['tb_module']
        
        if tb_module_name not in Extractor._meas_man_cache:
            # import once, cache forever
            meas_module = importlib.import_module(tb_module_name)
            OpampMeasMan = meas_module.OpampMeasMan
            Extractor._meas_man_cache[tb_module_name] = OpampMeasMan
        
        return Extractor._meas_man_cache[tb_module_name]

    def _resolve_size_map(self, full_params):
        """
        Resolve size_map into structured sizing using simple variable lookup (Titan-Killer #3).
        Avoids eval() and repeated string manipulation.
        
        Parameters:
        -----------
        full_params (dict): Full parameter dictionary.
        
        Returns:
        --------
        dict: Structured sizing dictionary.
        """
        if not self.size_map:
            return full_params
        
        structured_sizing = {}
        for comp, params in self.size_map.items():
            comp_props = {}
            for prop, val_expr in params.items():
                # simple string strip instead of eval
                clean_var = val_expr.replace('{{', '').replace('}}', '').strip()
                
                # direct lookup in full_params
                if clean_var in full_params:
                    comp_props[prop] = full_params[clean_var]
                else:
                    try:
                        comp_props[prop] = float(clean_var)
                    except (ValueError, TypeError):
                        comp_props[prop] = clean_var
            
            structured_sizing[comp] = comp_props
        
        return structured_sizing if structured_sizing else full_params

    def _apply_context(self, full_params):
        """
        apply required context values from sample to extractor state.
        """
        context_attr_map = {
            "fet_num": "fet_num",
            "vdd": "vdd",
            "vcm": "vcm",
            "tempc": "tempc",
            "is_hp": "is_hp",
            "n_state": "n_state",
            "p_state": "p_state",
            "cload_val": "cload_val",
        }

        for key, attr_name in context_attr_map.items():
            if key in full_params:
                setattr(self, attr_name, full_params[key])
            else:
                cur_val = getattr(self, attr_name)
                if cur_val is None:
                    raise ValueError(f"missing required context key: {key}")
                full_params[key] = cur_val

    def _vector_to_full_params(self, x):
        """
        Convert vector input into full parameter dict.

        Parameters:
        -----------
        x (np.ndarray): 1D vector aligned to self.opt_params.

        Returns:
        --------
        dict: Full parameter mapping for simulation.
        """
        assert len(x) == self.dim
        assert x.ndim == 1

        sample = x.copy()
        for i, param in enumerate(self.opt_params):
            if param.startswith('nB'):
                sample[i] = round(sample[i])

        full_params = {}
        opt_idx = 0
        for pname in self.params_id:
            if pname in self.opt_params:
                full_params[pname] = sample[opt_idx]
                opt_idx += 1

        # fill any leftover optimized params not present in params_id order
        while opt_idx < len(self.opt_params):
            pname = self.opt_params[opt_idx]
            if pname not in full_params:
                full_params[pname] = sample[opt_idx]
            opt_idx += 1

        return full_params

    def _dump_debug_specs(self, cur_specs):
        """
        Optionally dump specs when ASPECTOR_DEBUG_DUMP is enabled.

        Parameters:
        -----------
        cur_specs (dict): Current specs to dump for debugging.
        """
        if os.environ.get('ASPECTOR_DEBUG_DUMP', '').strip() not in {'1', 'true', 'TRUE', 'yes', 'YES'}:
            return

        import datetime

        debug_dir = os.path.join(os.path.dirname(__file__), '../../../..', 'debug_results_dump')
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        fname = f'final_results_dump_{timestamp}.txt'
        fpath = os.path.join(debug_dir, fname)

        def convert_keys_to_str(obj):
            if isinstance(obj, dict):
                return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_keys_to_str(i) for i in obj]
            return obj

        try:
            with open(fpath, 'w') as f:
                f.write(json.dumps(convert_keys_to_str(cur_specs), indent=2, default=str))
        except Exception as e:
            with open(fpath, 'w') as f:
                f.write(f'Failed to dump final cur_specs: {e}\n')
                f.write(str(cur_specs))

    def lookup(self, spec, goal_spec):
        """
        Calculates normalized spec deviation from targets.

        Parameters:
        -----------
        spec (list or np.ndarray): Measured specifications.
        goal_spec (list or np.ndarray): Target specifications.

        Returns:
        --------
        np.ndarray: Normalized deviations for each specification.
        """
        norm_specs = []
        for s, g in zip(spec, goal_spec):
            # range measurement
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) == 2:
                 # target as range
                 if isinstance(g, (list, tuple, np.ndarray)) and len(g) == 2:
                     # containment objective
                     val_min = (g[0] - s[0]) / (abs(g[0]) + abs(s[0]) + 1e-9)
                     val_max = (s[1] - g[1]) / (abs(g[1]) + abs(s[1]) + 1e-9)
                     norm_specs.append(min(val_min, val_max))
                 else:
                     # width objective
                     width_s = abs(s[1] - s[0])
                     width_g = float(g)
                     val = (width_s - width_g) / (abs(width_g) + abs(width_s) + 1e-9)
                     norm_specs.append(val)
            else:
                 # scalar measurement
                 s_val = float(s) if s is not None else 0.0
                 g_val = float(g)
                 val = (s_val - g_val) / (abs(g_val) + abs(s_val) + 1e-9)
                 norm_specs.append(val)

        return np.array(norm_specs)

    def _compute_smooth_penalty(self, spec_name, worst_valid_value=None, z_score_severity=5.0):
        """
        Compute a smooth penalty value for failed/missing specs.
        Prevents "discontinuity cliffs" that collapse GP lengthscales (Titan-Killer #1).
        
        Parameters:
        -----------
        spec_name (str): Name of the failed specification.
        worst_valid_value (float, optional): Worst observed valid value for this spec.
        z_score_severity (float): Z-score multiple for penalty (default 5.0 = 5σ beyond nominal).
        
        Returns:
        --------
        float: Smooth penalty value (exponential decay toward valid region).
        """
        # Conservative defaults for common specs
        penalty_defaults = {
            'power_w': 1e6,
            'integrated_noise_vrms': 1e7,
            'settling_time': 1e5,
            'settle_time_ns': 1e5,
            'vos_v': 1.0,
            'area': 1e5,
            'estimated_area_um2': 1e5,
        }
        
        if worst_valid_value is not None:
            # Use Z-score based penalty: worst_valid + 5σ
            # For minimize specs, 5σ above worst; for maximize, 5σ below
            if spec_name in ['power_w', 'integrated_noise_vrms', 'settling_time', 'settle_time_ns', 'vos_v', 'area', 'estimated_area_um2']:
                return worst_valid_value * (1.0 + z_score_severity)
            else:
                return worst_valid_value / (1.0 + z_score_severity)
        
        # Fallback to conservative defaults (much better than 1e9 cliff)
        return penalty_defaults.get(spec_name, 100.0)

    def reward(self, spec, goal_spec, specs_id, specs_weights):
        """
        Calculate penalty-based reward from measured specs.

        Parameters:
        -----------
        spec (list or np.ndarray): Measured specifications.
        goal_spec (list or np.ndarray): Target specifications.
        specs_id (list of str): Specification identifiers.
        specs_weights (list of float): Weights for each specification.

        Returns:
        --------
        float: Calculated reward based on specification deviations.
        """
        if not specs_id:
            return 0.0
            
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0
        
        # optimization direction
        minimize_specs = ["power_w", "integrated_noise_vrms", "settling_time", "settle_time_ns", "vos_v", "estimated_area_um2", "area"]
        
        for i, rel_spec in enumerate(rel_specs):
            s_name = specs_id[i]
            s_weight = specs_weights[i]
            
            if s_name in minimize_specs:
                if rel_spec > 0:
                    reward += s_weight * np.abs(rel_spec)
            else:
                if rel_spec < 0:
                    reward += s_weight * np.abs(rel_spec)

        return reward

    def _build_sizing_env_bias(self, full_params, final_sizing, iter_num=None):
        """
        Build sizing/env/bias dicts used in ids and result records.

        Parameters:
        -----------
        full_params (dict): Full set of parameters.
        final_sizing (dict): Final sizing parameters.
        iter_num (int, optional): Iteration number.

        Returns:
        --------
        dict: Dictionary containing sizing, environment, and bias information.
        """
        env_dict = {
            "vdd": self.vdd,
            "vcm": self.vcm,
            "tempc": self.tempc,
            "fet_num": self.fet_num,
            "is_hp": self.is_hp,
            "n_state": self.n_state,
            "p_state": self.p_state,
        }

        bias_dict = {}
        for k, v in full_params.items():
            if k not in env_dict:
                if k.lower().startswith('vbias') or k.lower().startswith('ibias') or 'bias' in k.lower():
                    bias_dict[k] = v

        sizing_data = {
            "topology_id": self.netlist_name,
            "netlist": self.netlist_name,
            "parameters": final_sizing,
            "sizing": final_sizing,
            "bias": bias_dict,
            "env": env_dict
        }
        if iter_num is not None:
            sizing_data['sim_id'] = iter_num
        return sizing_data

    def sim_key_for_params(self, full_params, final_sizing=None):
        """
        Compute deterministic sim_key for a given parametrization.

        Parameters:
        -----------
        full_params (dict): Full set of parameters.
        final_sizing (dict, optional): Final sizing parameters.

        Returns:
        --------
        str: Deterministic simulation key.
        """
        try:
            # rebuild final_sizing if not provided (use cached method)
            if final_sizing is None:
                final_sizing = self._resolve_size_map(full_params)

            sizing_data = {
                "netlist": self.netlist_name,
                "sizing": final_sizing,
                "bias": {k: v for k, v in full_params.items() if (k.lower().startswith('vbias') or k.lower().startswith('ibias') or 'bias' in k.lower())},
                "env": {
                    "vdd": full_params.get('vdd', self.vdd),
                    "vcm": full_params.get('vcm', self.vcm),
                    "tempc": full_params.get('tempc', self.tempc),
                    "fet_num": full_params.get('fet_num', self.fet_num),
                    "is_hp": full_params.get('is_hp', self.is_hp),
                    "n_state": full_params.get('n_state', self.n_state),
                    "p_state": full_params.get('p_state', self.p_state),
                },
            }
            key_source = json.dumps(sizing_data, sort_keys=True, separators=(',', ':'))
            return hashlib.sha1(key_source.encode('utf-8')).hexdigest()
        except Exception:
            return str(uuid.uuid4())

    def __call__(self, x, sim_id=None):
        """
        Run one simulation/evaluation call.

        Parameters:
        -----------
        x : numpy array or dict
            parameter vector of length self.dim.
        sim_id : int, optional
            explicit simulation id. If none, uses an internal Extractor counter.
        
        Returns:
        --------
        tuple: extractor output tuple used by logging and collection.
        """
        # preferred path: dict input from generator
        if isinstance(x, dict):
            full_params = x.copy()

            # ensure discrete params are int
            for pname, val in full_params.items():
                if pname.startswith("nB"):
                    full_params[pname] = int(round(val))
                    
        else:
            full_params = self._vector_to_full_params(x)

        # build measurement environment
        sim_env = None
        try:
            if not isinstance(self.config, dict):
                raise RuntimeError("Extractor requires a configuration dict, not a file path.")
            # Use cached OpampMeasMan to avoid 6.4M imports (Titan-Killer #2)
            OpampMeasMan = self._get_opamp_meas_man()
            sim_env = OpampMeasMan(self.config)
        except Exception as e:
            raise RuntimeError(f"Failed to build sim_env from config: {e}")

        if sim_id is not None:
             iter_num = sim_id
        else:
               Extractor._iter_counter += 1
               iter_num = Extractor._iter_counter
        
        print(f"\n {'='*60}")
        print(f" Iteration {iter_num}")
        print(f" {'='*60}")
        print(f"   [+] Running simulation...")

        # apply required context from sample
        self._apply_context(full_params)

        # gatekeeper is always enabled
        full_params["run_gatekeeper"] = 1

        # 0=dcop fail, 1=tier1 only, 2=full char
        sim_status = 0

        # always run tier 1 first
        full_params["run_full_char"] = 0
        param_val = [OrderedDict(full_params)]
        # Ensure Spectre wrapper can relocate artifacts into results_dir/raw when enabled.
        os.environ["ASPECTOR_RESULTS_DIR"] = str(self.results_dir)
        sim_env.ver_specs['results_dir'] = self.results_dir
        eval_result = sim_env.evaluate(param_val)

        if not eval_result or len(eval_result) == 0:
            eval_result = [(None, {}, 1)]

        try:
            sim_info = int(eval_result[0][2]) if len(eval_result[0]) > 2 else 0
        except Exception:
            sim_info = 0

        dcop_failed = False
        if sim_info != 0:
            print(f"   [!] Underlying simulation reported non-zero info={sim_info} — DCOP failure.")
            sim_status = 0
            bad_specs_dict = {}
            eval_result[0] = (eval_result[0][0], bad_specs_dict, sim_info)
            dcop_failed = True

        specs = eval_result[0][1]

        if not dcop_failed:
            real_vals = sum(1 for v in specs.values() if v is not None)
            print(f"   [GATEKEEPER] Tier 1: {real_vals}/{len(specs)} specs extracted")
            if real_vals == 0:
                print(f"   [!] Tier 1 returned NO data — Spectre likely failed.")
                sim_status = 0
            else:
                sim_status = 0
        else:
            print("   [i] Skipping gatekeeper checks due to DCOP failure.")

        # gatekeeper and tier-2 logic is skipped if dcop already failed
        if not dcop_failed:
            region_MM = specs.get('zregion_of_operation_MM', {})
            ids_MM = specs.get('zzids_MM', {})
            if not isinstance(region_MM, dict):
                region_MM = {}
            if not isinstance(ids_MM, dict):
                ids_MM = {}
            mm_names = sorted(set(region_MM.keys()) | set(ids_MM.keys()))

            ops_good = True
            if not mm_names or not region_MM:
                ops_good = False
            else:
                for mm in mm_names:
                    region = region_MM.get(mm)
                    ids = ids_MM.get(mm)
                    if region is None or region == 0.0 or region == 4.0:
                        ops_good = False
                        break
                    try:
                        if (ids is None) or float(ids) <= 0.0:
                            ops_good = False
                            break
                    except Exception:
                        ops_good = False
                        break

            if not ops_good:
                print(f"   [-] Transistor ops bad — skipping Tier 2.")

            if ops_good:
                sim_status = 2
                full_params["run_full_char"] = 1
                param_val = [OrderedDict(full_params)]
                sim_env.ver_specs['results_dir'] = self.results_dir
                eval_result = sim_env.evaluate(param_val)
            else:
                if real_vals > 0:
                    sim_status = 1
                else:
                    sim_status = 0
                bad_specs_dict = {}
                for k, v in specs.items():
                    if v is None:
                        continue
                    if k.startswith('z') and isinstance(v, dict):
                        if k not in bad_specs_dict or not isinstance(bad_specs_dict.get(k), dict):
                            bad_specs_dict[k] = {}
                        for comp, val in v.items():
                            bad_specs_dict[k][comp] = val
                    else:
                        bad_specs_dict[k] = v
                eval_result[0] = (eval_result[0][0], bad_specs_dict)
        else:
            pass
        
        if not eval_result or len(eval_result) == 0:
            print(f"   [!] Simulation returned empty results.")
            bad_specs_dict = {}
            eval_result = [(None, bad_specs_dict)]
        

        cur_specs = OrderedDict(sorted(eval_result[0][1].items(), key=lambda k:k[0]))

        self._dump_debug_specs(cur_specs)

        # keep failed/missing values as nan
        for key, val in list(cur_specs.items()):
            if val is None:
                cur_specs[key] = float('nan')

        # formatted spec output
        is_valid = (sim_status == 2)
        status_tag = "PASS" if is_valid else "FAIL"
        print(f"   [+] Measurement complete — [{status_tag}]")
        sim_status_label = {0: 'DCOP_FAIL', 1: 'TIER1_ONLY', 2: 'FULL_CHAR'}.get(sim_status, 'UNKNOWN')
        print(f"   [i] sim_status: {sim_status} ({sim_status_label})")
        print(f"   {'-'*50}")

        op_keys = [k for k in cur_specs if k.startswith('z')]
        main_keys = [k for k in cur_specs if not k.startswith('z') and k != 'valid']

        if main_keys:
            col_w = max(len(k) for k in main_keys) + 2
            for k in main_keys:
                v = cur_specs[k]
                if isinstance(v, float):
                    print(f"   {k:<{col_w}} {v:>14.4g}")
                else:
                    print(f"   {k:<{col_w}} {str(v):>14}")

        if op_keys:
            sample_dict = cur_specs.get(op_keys[0], {})
            if isinstance(sample_dict, dict) and sample_dict:
                mm_names = sorted(sample_dict.keys())
                param_labels = []
                for ok in op_keys:
                    label = ok.lstrip('z')
                    if label.endswith('_MM'):
                        label = label[:-3]
                    param_labels.append(label)

                cap_params = {'cgg', 'cgs', 'cdd', 'cgd', 'css'}
                core_labels = [lbl for lbl in param_labels if lbl not in cap_params]
                cap_labels = [lbl for lbl in param_labels if lbl in cap_params]

                label_to_key = {}
                for ok in op_keys:
                    lab = ok.lstrip('z')
                    if lab.endswith('_MM'):
                        lab = lab[:-3]
                    label_to_key[lab] = ok

                header_core = f"   {'FET':<6}" + "".join(f"{lbl:>12}" for lbl in core_labels)
                print(f"\n   Operating Points:")
                print(header_core)
                print(f"   {'-'*(6 + 12 * len(core_labels))}")
                for mm in mm_names:
                    row = f"   {mm:<6}"
                    for lbl in core_labels:
                        ok = label_to_key.get(lbl)
                        val = cur_specs.get(ok, {}).get(mm, 0.0) if ok else 0.0
                        if isinstance(val, float):
                            row += f"{val:>12.4g}"
                        else:
                            row += f"{str(val):>12}"
                    print(row)

                if cap_labels:
                    header_cap = f"   {'FET':<6}" + "".join(f"{lbl:>12}" for lbl in cap_labels)
                    print(f"\n   Capacitances:")
                    print(header_cap)
                    print(f"   {'-'*(6 + 12 * len(cap_labels))}")
                    for mm in mm_names:
                        row = f"   {mm:<6}"
                        for lbl in cap_labels:
                            ok = label_to_key.get(lbl)
                            val = cur_specs.get(ok, {}).get(mm, 0.0) if ok else 0.0
                            if isinstance(val, float):
                                row += f"{val:>12.4g}"
                            else:
                                row += f"{str(val):>12}"
                        print(row)
        print(f"   {'-'*50}")
        
        # reward calculation for optimizer pathways
        reward_input = []
        if self.specs_id:
            for s_name in self.specs_id:
                val = cur_specs.get(s_name)
                if val is None:
                    # Smooth penalty instead of discontinuous cliff (Titan-Killer #1)
                    val = self._compute_smooth_penalty(s_name, worst_valid_value=None)
                reward_input.append(val)
            
            reward1 = self.reward(reward_input, self.specs_ideal, self.specs_id, self.specs_weights)
        else:
            reward1 = 0.0

        # build structured sizing from map
        final_sizing = self._resolve_size_map(full_params)

        sizing_data = self._build_sizing_env_bias(full_params, final_sizing, iter_num)

        # remove env keys from sizing section
        for k in ["vdd", "vcm", "tempc", "fet_num", "is_hp", "n_state", "p_state"]:
            if k in sizing_data["sizing"]:
                del sizing_data["sizing"][k]

        # remove unwanted components
        comps_to_remove = []
        for comp in sizing_data["sizing"]:
            if comp.startswith("Rshunt") or comp.startswith("Rsw") or comp.startswith("R_unity") or comp.startswith("Rsrc_"):
                comps_to_remove.append(comp)
        
        for comp in comps_to_remove:
            del sizing_data["sizing"][comp]

        # consolidate Cload
        if "cload_val" in full_params:
            sizing_data["sizing"]["Cload"] = {"c": full_params["cload_val"]}

        # generate deterministic simulation key
        try:
            key_source = json.dumps({
                "netlist": self.netlist_name,
                "sizing": sizing_data["sizing"],
                "bias": sizing_data["bias"],
                "env": sizing_data["env"]
            }, sort_keys=True, separators=(',', ':'))
            sim_key = hashlib.sha1(key_source.encode('utf-8')).hexdigest()
        except Exception:
            sim_key = str(uuid.uuid4())

        raw_specs = eval_result[0][1] if eval_result and len(eval_result) > 0 and len(eval_result[0]) > 1 else {}
        
        main_specs = {}
        op_points = {}
        
        def merge_op_data(target_dict, param_name, data_dict):
            """
            Merge operating point data into the target dictionary.

            Parameters:
            -----------
            target_dict (dict): Dictionary to merge data into.
            param_name (str): Name of the parameter.
            data_dict (dict or list): Data to merge.
            """
            if isinstance(data_dict, dict):
                for comp_name, val in data_dict.items():
                    if comp_name not in target_dict:
                        target_dict[comp_name] = {}
                    try:
                        if isinstance(val, (int, float)):
                            target_dict[comp_name][param_name] = float(val)
                        else:
                            target_dict[comp_name][param_name] = val
                    except Exception:
                        target_dict[comp_name][param_name] = val
            elif isinstance(data_dict, list):
                 pass

        for k, v in raw_specs.items():
            if k.startswith("z"):
                clean_k = k.lstrip('z')
                if clean_k.endswith('_MM'):
                    clean_k = clean_k[:-3]

                merge_op_data(op_points, clean_k, v)
            else:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        main_specs[k] = float(v)
                    except Exception:
                        main_specs[k] = v
                elif isinstance(v, tuple):
                    lst = []
                    for x in v:
                        try:
                            lst.append(float(x) if isinstance(x, (int, float)) else x)
                        except Exception:
                            lst.append(x)
                    main_specs[k] = lst
                else:
                    main_specs[k] = v

            # sanitize parameters for per-sim jsons
        parameters = dict(sizing_data.get("sizing", {}))
        for _k in ("R_cmfb_pole", "C_cmfb_pole"):
            parameters.pop(_k, None)

        simulation_result = {
            "id": sim_key,
            "sim_id": sim_key,
            "topology_id": 1,
            "netlist": self.netlist_name,
            "parameters": parameters,
            "bias": sizing_data["bias"],
            "env": sizing_data["env"],
            "specs": main_specs,
            "operating_points": op_points,
            "sim_status": sim_status
        }

        # non-mass-collection mode writes per-sim jsons
        result_fname = os.path.join(self.results_dir, f"{sim_key}.json")
        lock_fname = result_fname + ".lock"

        try:
            if os.path.exists(result_fname):
                try:
                    with open(result_fname, 'r') as rf:
                        existing = json.load(rf)
                    existing_main_specs = existing.get('specs', main_specs)
                    return reward1, existing_main_specs, existing
                except Exception:
                    pass

            # create lock atomically
            lock_fd = None
            try:
                lock_fd = os.open(lock_fname, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(lock_fd, f"pid:{os.getpid()}\nstart:{time.time()}\n".encode('utf-8'))
            except OSError as e:
                lock_fd = None
                if e.errno == errno.EEXIST:
                    # another process is running this parametrization
                    sim_ctrl = self.config.get('sim_control', {}) if isinstance(self.config, dict) else {}
                    WAIT_TIMEOUT = float(sim_ctrl.get('wait_timeout', 30.0))
                    POLL = float(sim_ctrl.get('poll_interval', 0.2))
                    STALE_LOCK_AGE = float(sim_ctrl.get('stale_lock_age', 600.0))

                    # remove stale lock and retry once
                    try:
                        mtime = os.path.getmtime(lock_fname)
                        age = time.time() - mtime
                        if age > STALE_LOCK_AGE:
                            try:
                                os.remove(lock_fname)
                                lock_fd = os.open(lock_fname, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                                os.write(lock_fd, f"pid:{os.getpid()}\nstart:{time.time()}\n".encode('utf-8'))
                            except Exception:
                                lock_fd = None
                    except Exception:
                        pass

                    waited = 0.0
                    while waited < WAIT_TIMEOUT:
                        if os.path.exists(result_fname):
                            try:
                                with open(result_fname, 'r') as rf:
                                    existing = json.load(rf)
                                existing_main_specs = existing.get('specs', main_specs)
                                if 'specs' not in existing:
                                    try:
                                        existing['specs'] = existing_main_specs
                                    except Exception:
                                        existing['specs'] = {}
                                return reward1, existing_main_specs, existing
                            except Exception:
                                pass
                        time.sleep(POLL)
                        waited += POLL

                        # timeout waiting for result; return fail payload
                    bad_specs = {}
                    fail_result = {
                        "id": sim_key,
                        "sim_id": sim_key,
                        "netlist": self.netlist_name,
                        "parameters": sizing_data.get("parameters", sizing_data.get("sizing", {})),
                        "bias": sizing_data.get("bias", {}),
                        "env": sizing_data.get("env", {}),
                        "specs": bad_specs,
                        "operating_points": {},
                        "sim_status": 0
                    }
                    return 0.0, bad_specs, fail_result

            # lock owner path
            try:
                if self.mode == "mass_collection":
                    return reward1, main_specs, simulation_result
                else:
                    tmp_path = result_fname + ".tmp"
                    with open(tmp_path, 'w') as tf:
                        json.dump(simulation_result, tf, indent=2)
                    os.replace(tmp_path, result_fname)
                    print(f"   [+] Exported result: {sim_key}")
                    return reward1, main_specs
            finally:
                try:
                    if lock_fd:
                        os.close(lock_fd)
                    if os.path.exists(lock_fname):
                        os.remove(lock_fname)
                except Exception:
                    pass
        except Exception as e:
            # keep worker alive on unexpected errors
            bad_specs = {}
            fail_result = {
                "id": sim_key,
                "sim_id": sim_key,
                "netlist": self.netlist_name,
                "parameters": sizing_data["sizing"],
                "bias": sizing_data["bias"],
                "env": sizing_data["env"],
                "specs": bad_specs,
                "operating_points": {},
                "sim_status": 0
            }
            return 0.0, bad_specs, fail_result
