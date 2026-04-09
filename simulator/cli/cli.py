"""
cli.py

Author: natelgrw
Last Edited: 01/24/2026

Command line interface for Aspectryx Core.
Provides interactive setup for running circuit optimization pipelines, supporting parallel execution.
"""

import os
import sys
import multiprocessing as mp
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from simulator.compute.runner import run_parallel_simulations
from simulator.compute.collector import DataCollector
from algorithms.sobol.generator import SobolSizingGenerator
from algorithms.turbo_m import TurboMSizingGenerator
from simulator.eval_engines.utils.design_reps import extract_sizing_map
from simulator.eval_engines.spectre.configs.config_env import EnvironmentConfig
from simulator.eval_engines.extractor.extractor import Extractor, extract_parameter_names, classify_opamp_type
import torch


# ===== CLI Styling and Utilities ===== #


class Style:
    CHECK = "[+]"
    X = "[!]"
    INFO = "..."
    ARROW = ">>"
    LINE = "-" * 70
    DOUBLE_LINE = "=" * 70

def clear_screen():
    """
    Clears the terminal screen.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """
    Prints the header for the CLI.
    """
    print()
    print(Style.DOUBLE_LINE)
    print("       TITAN FOUNDATION MODEL - CIRCUIT OPTIMIZATION PIPELINE       ".center(70))
    print(Style.DOUBLE_LINE)
    print()

def print_section(title):
    """
    Prints a section header for the CLI.

    Parameters:
    -----------
    title (str): The title of the section to print.
    """
    print(f"\n{Style.LINE}")
    print(f" {title.upper()} ".center(70))
    print(f"{Style.LINE}\n")

def print_success(message):
    """
    Prints a success message.

    Parameters:
    -----------
    message (str): The message to print.
    """
    print(f" {Style.CHECK} {message}")

def print_info(message):
    """
    Prints an informational message.

    Parameters:
    -----------
    message (str): The message to print.
    """

    print(f" {Style.INFO} {message}")

def print_error(message):
    """
    Prints an error message.

    Parameters:
    -----------
    message (str): The message to print.
    """
    print(f" {Style.X} Error: {message}")


# ===== Optimization Persona Configurations ===== #


PERSONA_CONFIGS = {
    1: {
        "name": "SPEED",
        "weights": {
            'ugbw_hz': 30.0,
            'slew_rate_v_us': 24.0,
            'settle_time_small_ns': 14.0,
            'settle_time_large_ns': 12.0,
            'pm_deg': 8.0,
            'gain_ol_dc_db': 5.0,
            'power_w': 4.0,
            'output_voltage_swing_range_v': 3.0,
            '_pm_deg_target': 62.5,
            '_pm_deg_range': 2.5,
        },
    },
    2: {
        "name": "PRECISION",
        "weights": {
            'vos_v': 25.0,
            'thd_db': 20.0,
            'integrated_noise_vrms': 20.0,
            'gain_ol_dc_db': 10.0,
            'output_voltage_swing_range_v': 10.0,
            'cmrr_dc_db': 7.0,
            'psrr_dc_db': 7.0,
            'settle_time_small_ns': 1.0,
        },
    },
    3: {
        "name": "EFFICIENCY",
        "weights": {
            'power_w': 45.0,
            'estimated_area_um2': 20.0,
            'integrated_noise_vrms': 10.0,
            'gain_ol_dc_db': 8.0,
            'ugbw_hz': 7.0,
            'pm_deg': 5.0,
            'settle_time_small_ns': 3.0,
            'settle_time_large_ns': 2.0,
        },
    },
    4: {
        "name": "COMPACTNESS",
        "weights": {
            'estimated_area_um2': 45.0,
            'power_w': 20.0,
            'gain_ol_dc_db': 10.0,
            'pm_deg': 10.0,
            'ugbw_hz': 5.0,
            'output_voltage_swing_range_v': 5.0,
            'settle_time_small_ns': 3.0,
            'settle_time_large_ns': 2.0,
            '_pm_deg_target': 67.5,
            '_pm_deg_range': 7.5,
        },
    },
    5: {
        "name": "BALANCED",
        "weights": {
            'gain_ol_dc_db': 15.0,
            'ugbw_hz': 15.0,
            'pm_deg': 10.0,
            'power_w': 12.0,
            'vos_v': 10.0,
            'integrated_noise_vrms': 10.0,
            'output_voltage_swing_range_v': 10.0,
            'slew_rate_v_us': 8.0,
            'thd_db': 5.0,
            'estimated_area_um2': 3.0,
            'settle_time_small_ns': 1.0,
            'settle_time_large_ns': 1.0,
        },
    },
    6: {
        "name": "ROBUSTNESS",
        "weights": {
            'pm_deg': 16.0,
            'gain_ol_dc_db': 12.0,
            'cmrr_dc_db': 8.0,
            'psrr_dc_db': 8.0,
            'output_voltage_swing_range_v': 10.0,
            'power_w': 8.0,
            'vos_v': 8.0,
            'integrated_noise_vrms': 8.0,
            'settle_time_small_ns': 8.0,
            'settle_time_large_ns': 6.0,
            'slew_rate_v_us': 6.0,
            'thd_db': 2.0,
            'estimated_area_um2': 8.0,
            '_pm_deg_target': 70.0,
            '_pm_deg_range': 10.0,
        },
    },
    7: {
        "name": "LINEARITY",
        "weights": {
            'thd_db': 30.0,
            'integrated_noise_vrms': 20.0,
            'output_voltage_swing_range_v': 15.0,
            'gain_ol_dc_db': 8.0,
            'vos_v': 8.0,
            'slew_rate_v_us': 6.0,
            'ugbw_hz': 5.0,
            'pm_deg': 4.0,
            'cmrr_dc_db': 2.0,
            'psrr_dc_db': 2.0,
        },
    },
    8: {
        "name": "LOW_HEADROOM",
        "weights": {
            'output_voltage_swing_range_v': 25.0,
            'vos_v': 15.0,
            'gain_ol_dc_db': 12.0,
            'ugbw_hz': 10.0,
            'pm_deg': 10.0,
            'power_w': 10.0,
            'thd_db': 6.0,
            'integrated_noise_vrms': 6.0,
            'settle_time_small_ns': 3.0,
            'settle_time_large_ns': 3.0,
            '_pm_deg_target': 60.0,
            '_pm_deg_range': 10.0,
        },
    },
    9: {
        "name": "STARTUP_RELIABILITY",
        "weights": {
            'settle_time_small_ns': 22.0,
            'settle_time_large_ns': 18.0,
            'pm_deg': 14.0,
            'slew_rate_v_us': 12.0,
            'gain_ol_dc_db': 8.0,
            'power_w': 8.0,
            'integrated_noise_vrms': 6.0,
            'vos_v': 6.0,
            'ugbw_hz': 4.0,
            'estimated_area_um2': 2.0,
            '_pm_deg_target': 65.0,
            '_pm_deg_range': 10.0,
        },
    },
    10: {
        "name": "DRIVE_LOAD",
        "weights": {
            'slew_rate_v_us': 22.0,
            'output_voltage_swing_range_v': 20.0,
            'ugbw_hz': 18.0,
            'settle_time_large_ns': 12.0,
            'settle_time_small_ns': 8.0,
            'pm_deg': 8.0,
            'gain_ol_dc_db': 6.0,
            'power_w': 4.0,
            'thd_db': 2.0,
            '_pm_deg_target': 62.5,
            '_pm_deg_range': 7.5,
        },
    },
}


# ===== CLI Input Handling Functions ===== #


def get_persona_config(persona_id):
    """
    Return persona name and weight dictionary.

    Parameters:
    -----------
    persona_id (int): Persona identifier in [1, 10].

    Returns:
    --------
    tuple[str, dict]: Persona name and corresponding weights.
    """
    cfg = PERSONA_CONFIGS.get(persona_id)
    if cfg is None:
        raise ValueError(f"Unknown persona id: {persona_id}")
    return cfg["name"], cfg["weights"]

def get_valid_int(prompt, min_val, max_val, default):
    """
    Prompt for an integer within a range.

    Parameters:
    -----------
    prompt (str): Prompt label shown to the user.
    min_val (int): Minimum allowed value.
    max_val (int): Maximum allowed value.
    default (int): Default value if user input is empty.

    Returns:
    --------
    int: User-selected integer within [min_val, max_val].
    """
    while True:
        user_input = input(f" {Style.ARROW} {prompt} [{default}]: ").strip()
        
        if not user_input:
            return default
            
        try:
            val = int(user_input)
            if min_val <= val <= max_val:
                return val
            else:
                print_error(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print_error("Invalid integer.")


def get_netlist_selection():
    """
    Returns a list of tuples: [(netlist_name, full_path), ...]
    Allows user to select a custom directory or use the default.
    """
    print_section("Netlist Selection")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    default_path = os.path.join(project_root, "topologies")
    
    print(f" Default Directory: {default_path}")
    print(" (Press ENTER to use default, or paste a custom folder path)")
    
    custom_path = input(f" {Style.ARROW} Directory: ").strip()
    
    base_path = default_path
    if custom_path:
        expanded_path = os.path.expanduser(custom_path)
        expanded_path = os.path.abspath(expanded_path)
        
        if os.path.isdir(expanded_path):
            base_path = expanded_path
            print_success(f"Using directory: {base_path}")
        else:
            print_error(f"Directory not found: {custom_path}. Reverting to default.")

    available_files = []
    if os.path.exists(base_path):
        available_files = sorted([f for f in os.listdir(base_path) if f.endswith('.scs')])
        
        if not available_files:
            print_error(f"No .scs files found in {base_path}")
            return []
            
        print("\n Available Netlists:")
        for i, f in enumerate(available_files):
            print(f"   {i+1}. {f[:-4]}")
        print(f"   {len(available_files)+1}. [BATCH] Run ALL Netlists in folder")
        print()
    else:
        print_error(f"Path does not exist: {base_path}")
        return get_netlist_selection()

    while True:
        prompt = f" {Style.ARROW} Enter selection (Number or Name): "
        user_input = input(prompt).strip()
        
        if not user_input:
            continue
            
        # check for file selection
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(available_files):
                fname = available_files[idx]
                return [(fname[:-4], os.path.join(base_path, fname))]
            elif idx == len(available_files):
                print_info(f"Selected Batch Mode: {len(available_files)} netlists")
                return [(f[:-4], os.path.join(base_path, f)) for f in available_files]
            else:
                 print_error("Invalid selection number")
                 continue
        
        # fallback to name string match
        if not user_input.endswith('.scs'):
            target_f = user_input + ".scs"
        else:
            target_f = user_input
            
        full_path = os.path.join(base_path, target_f)
            
        if os.path.exists(full_path):
            print_success(f"Netlist loaded: {user_input}")
            return [(user_input.replace('.scs',''), full_path)]
        else:
            print_error(f"File not found: {target_f}")


def get_turbo_mode(use_turbo):
    """
    Prompts for Blind vs Sight (Warm Start from Sobol) mode.

    Parameters:
    -----------
    use_turbo (bool): Whether TuRBO mode is active.

    Returns:
    --------
    str: 'blind' or 'sight'.
    """
    if not use_turbo:
        return "blind"

    print("\n TuRBO Initialization Mode:")
    print("   1. Blind (Default) - Start from scratch")
    print("   2. Sight (Warm Start) - Load Sobol data from results directory")
    
    while True:
        sel = input(f" {Style.ARROW} Enter selection [1]: ").strip()
        if not sel:
            sel = "1"

        if sel == "1":
            return "blind"
        if sel == "2":
            print_success("Sight mode selected. Sobol parquets will be loaded from the results directory.")
            return "sight"
        print_error("Invalid selection.")


def _parquet_to_specs(df):
    """
    Reconstruct spec dicts from a Parquet DataFrame (out_ columns).
    Returns list[dict] aligned with df rows.

    Parameters:
    -----------
    df (pd.DataFrame): DataFrame containing 'out_' columns for specs.

    Returns:
    --------
    list[dict]: List of spec dictionaries for each row, with keys stripped of 'out_' prefix.
                Special handling for 'output_voltage_swing_range_v' to reconstruct as tuple.
    """
    spec_cols = [c for c in df.columns if c.startswith('out_')]

    # detect swing columns
    swing_min = 'out_output_voltage_swing_min_v' if 'out_output_voltage_swing_min_v' in df.columns else None
    swing_max = 'out_output_voltage_swing_max_v' if 'out_output_voltage_swing_max_v' in df.columns else None
    
    specs_list = []
    for _, row in df.iterrows():
        s = {}
        s['valid'] = bool(row.get('valid', False))
        for col in spec_cols:
            if 'output_voltage_swing_range_v' in col or 'output_voltage_swing_min_v' in col or 'output_voltage_swing_max_v' in col:
                continue
            key = col[4:]
            if key == 'area':
                key = 'estimated_area_um2'
            val = row[col]
            if pd.notna(val):
                s[key] = float(val)
        # Reconstruct swing tuple
        if swing_min and swing_max and pd.notna(row[swing_min]) and pd.notna(row[swing_max]):
            s['output_voltage_swing_range_v'] = (float(row[swing_min]), float(row[swing_max]))
        specs_list.append(s)
    return specs_list


def _find_sobol_parquets(netlist_name_base, results_dir):
    """
    Scan the Sobol results directory for Parquet files matching this netlist.

    Parameters:
    -----------
    netlist_name_base (str): Base netlist name used as parquet filename prefix.
    results_dir (str): Netlist-specific results directory.

    Returns:
    --------
    list[str]: Sorted full paths for matching parquet files; empty if none found.
    """
    sobol_dir = os.path.join(results_dir, "sobol")
    if not os.path.isdir(sobol_dir):
        return []

    prefix = f"{netlist_name_base}_"
    found = []
    for f in sorted(os.listdir(sobol_dir)):
        if f.endswith('.parquet') and f.startswith(prefix):
            found.append(os.path.join(sobol_dir, f))
    return found


def run_optimization_task(netlist_name_base, scs_file_path, run_config):
    """
    Execute the optimization pipeline for a single netlist.

    This function is the orchestration layer for setup, simulation dispatch,
    checkpointing, and data logging. The optimization objective itself is
    defined by the active weight dictionary passed into `TurboMSizingGenerator`
    (consumed in `scalarize_specs` inside `algorithms/turbo_m/turbo_m.py`).

    Parameters:
    -----------
    netlist_name_base (str): Base name of the netlist (without .scs).
    scs_file_path (str): Full path to the .scs netlist file.
    run_config (dict): Configuration dictionary containing target parameters.
    """
    print_section(f"Processing: {netlist_name_base}")
    
    # unpack config
    n_workers = run_config['n_workers']
    adaptive_workers = run_config.get('adaptive_workers', False)
    use_turbo = run_config['use_turbo']
    use_mass_collection = run_config['use_mass_collection']
    

    # pipeline execution setup
    results_dir = os.path.join(run_config['output_dir'], netlist_name_base)
    os.makedirs(results_dir, exist_ok=True)

    # subdirectories for sobol, turbo_m
    sobol_dir = os.path.join(results_dir, "sobol")
    turbo_dir = os.path.join(results_dir, "turbo_m")

    # ensure directories exist
    os.makedirs(sobol_dir, exist_ok=True)
    os.makedirs(turbo_dir, exist_ok=True)

    collector = None
    sobol_state = os.path.join(sobol_dir, "sobol_state.txt")

    if use_mass_collection:
        if use_turbo:
            collector = DataCollector(output_dir=turbo_dir, buffer_size=1000, parquet_name=f"{netlist_name_base}_turbo_m.parquet")
        else:
            collector = DataCollector(output_dir=sobol_dir, buffer_size=1000, parquet_name=f"{netlist_name_base}_sobol.parquet")

    size_map = extract_sizing_map(scs_file_path)

    # extract parameters
    params_id = extract_parameter_names(scs_file_path)
    ignored_params = ['fet_num', 'vdd', 'vcm', 'tempc', 'cload_val', 'loop_mode']
    sizing_params_for_gen = [p for p in params_id if p not in ignored_params]
    
    generator = SobolSizingGenerator(sizing_params_for_gen, seed=None, topology=netlist_name_base)

    required_context_keys = {"fet_num", "vdd", "vcm", "tempc", "is_hp", "n_state", "p_state", "cload_val"}

    def _validate_sample_context(samples, stage_name):
        if not samples:
            return
        missing = []
        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                missing.append((idx, "<not-a-dict>"))
                continue
            miss = sorted(k for k in required_context_keys if k not in sample)
            if miss:
                missing.append((idx, ",".join(miss)))
                if len(missing) >= 3:
                    break
        if missing:
            details = "; ".join([f"idx={i} missing={m}" for i, m in missing])
            raise RuntimeError(f"{stage_name}: missing required context keys for extractor ({details})")
    
    opt_dim = generator.dim if use_turbo else generator.dim

    opamp_type = classify_opamp_type(scs_file_path)
    
    full_lb = [-1e9] * len(params_id)
    full_ub = [ 1e9] * len(params_id)
    
    config_env = EnvironmentConfig(scs_file_path, opamp_type, {}, params_id, full_lb, full_ub, results_dir=turbo_dir if use_turbo else sobol_dir)
    config_dict = config_env.get_config_dict()
    
    # default optimization goals
    if not use_turbo:
        specs_id = ["gain_ol_dc_db", "ugbw_hz", "pm_deg", "power_w", "vos_v"]
        specs_ideal = [0.0] * len(specs_id) 
        specs_weights = [1.0, 1.0, 10.0, 10.0, 10.0]
        
        extractor = Extractor(
            dim=len(params_id),
            opt_params=params_id,
            params_id=params_id,
            specs_id=specs_id,            
            specs_ideal=specs_ideal,
            specs_weights=specs_weights,
            config=config_dict,
            results_dir=sobol_dir, 
            netlist_name=netlist_name_base,
            size_map=size_map,
            mode="mass_collection"
        )

    n_max_evals = run_config['n_max_evals']
    interrupted = False

    try:
        if not use_turbo:
            # Sobol mode
            print(f" {Style.INFO} Mode: Sobol Exploration (One-shot)")
            start_idx = 0
            if os.path.exists(sobol_state):
                try:
                    with open(sobol_state, 'r') as f:
                        start_idx = int(f.read().strip())
                    print_info(f"Resuming Sobol sequence from index {start_idx}")
                except Exception as e:
                    print_error(f"Failed to read Sobol state: {e}")
            samples = generator.generate(n_max_evals, start_idx=start_idx)
            _validate_sample_context(samples, "sobol generation")
            # inject flags
            for s in samples:
                s['run_gatekeeper'] = 1
                s['run_full_char'] = 1
            
            # pre-check to catch major issues before running large batch
            print_info("Running pre-flight simulation check (1 sample, no multiprocessing)...")
            try:
                preflight_result = extractor(samples[0], sim_id=0)
                if preflight_result is None:
                    print_error("Pre-flight returned None. Spectre may not be working.")
                else:
                    n_vals = len(preflight_result) if isinstance(preflight_result, tuple) else 0
                    print_success(f"Pre-flight passed! Extractor returned {n_vals}-tuple.")
                    if n_vals >= 2 and isinstance(preflight_result[1], dict):
                        n_specs = len([k for k, v in preflight_result[1].items() if v is not None])
                        print_info(f"  Extracted {n_specs} non-null specs from pre-flight sample.")
            except Exception as e:
                import traceback
                print_error(f"Pre-flight FAILED: {e}")
                traceback.print_exc()
                print_error("Fix the above error before running the batch. Aborting.")
                return
            
            completed = 0
            for (completed, total, elapsed, data) in run_parallel_simulations(samples, extractor, n_workers, adaptive=adaptive_workers):
                rate = completed / elapsed if elapsed > 0 else 0
                percent = (completed / total) * 100
                bar = '#' * int(30 * completed // total) + '-' * (30 - int(30 * completed // total))
                print(f" [{bar}] {percent:5.1f}% | {completed}/{total} | Rate: {rate:4.1f} sim/s", end='\r')
                if data and use_mass_collection and collector:
                    idx, result_val = data
                    if len(result_val) == 3:
                        full_res = result_val[2] or {}
                        flat_config = samples[idx]
                        specs_to_log = None
                        if isinstance(full_res, dict):
                            specs_to_log = full_res.get('specs')
                        if specs_to_log is None and len(result_val) >= 2:
                            specs_to_log = result_val[1] or {}
                        if specs_to_log is None:
                            specs_to_log = {}

                        collector.log(
                            flat_config,
                            specs_to_log,
                            meta={
                                'sim_id': full_res.get('id', None),
                                'sim_status': full_res.get('sim_status', -1) if isinstance(full_res, dict) else -1,
                                'algorithm': 'sobol',
                                'netlist_name': netlist_name_base
                            },
                            operating_points=(full_res.get('operating_points') if isinstance(full_res, dict) else None)
                        )
                    else:
                        idx, result_val = data
                        flat_config = samples[idx]
                        specs_to_log = result_val[1] if isinstance(result_val, (list, tuple)) and len(result_val) > 1 else {}
                        operating_points = None
                        if isinstance(specs_to_log, dict):
                            op_temp = {}
                            for kk, vv in specs_to_log.items():
                                if isinstance(kk, str) and kk.startswith('z') and isinstance(vv, dict):
                                    clean_k = kk.lstrip('z')
                                    if clean_k.endswith('_MM'):
                                        clean_k = clean_k[:-3]
                                    for comp, val in vv.items():
                                        if comp not in op_temp:
                                            op_temp[comp] = {}
                                        try:
                                            if isinstance(val, (int, float)):
                                                op_temp[comp][clean_k] = float(val)
                                            else:
                                                op_temp[comp][clean_k] = val
                                        except Exception:
                                            op_temp[comp][clean_k] = val
                            if op_temp:
                                operating_points = op_temp

                        collector.log(
                            flat_config,
                            specs_to_log or {},
                            meta={
                                'sim_id': None,
                                'sim_status': -1,
                                'algorithm': 'sobol',
                                'netlist_name': netlist_name_base
                            },
                            operating_points=operating_points
                        )
                
                # save Sobol state in batches of 1000 to survive hard crashes (atomic write)
                if completed % 1000 == 0:
                    try:
                        with open(sobol_state + '.tmp', 'w') as f:
                            f.write(str(start_idx + completed))
                        os.replace(sobol_state + '.tmp', sobol_state)
                    except Exception:
                        pass

            # save final state (atomic write)
            try:
                with open(sobol_state + '.tmp', 'w') as f:
                    f.write(str(start_idx + n_max_evals))
                os.replace(sobol_state + '.tmp', sobol_state)
                print_info(f"Saved Sobol state at index {start_idx + n_max_evals}")
            except Exception as e:
                print_error(f"Failed to save Sobol state: {e}")
        else:
            # TuRBO-M mode
            personas_to_run = run_config.get('personas_to_run', [5])
            
            for p_idx in personas_to_run:
                p_name, objective_weights = get_persona_config(p_idx)
                    
                print(f"\n {Style.DOUBLE_LINE}")
                print(f" {Style.INFO} Starting TuRBO-M Optimization Loop - Persona: {p_name}")
                print(f" {Style.DOUBLE_LINE}")
                
                specs_id = list(objective_weights.keys())
                specs_ideal = [0.0] * len(specs_id) 
                specs_weights_list = list(objective_weights.values())
                
                extractor = Extractor(
                    dim=len(params_id),
                    opt_params=params_id,
                    params_id=params_id,
                    specs_id=specs_id,            
                    specs_ideal=specs_ideal,
                    specs_weights=specs_weights_list,
                    config=config_dict,
                    results_dir=turbo_dir, 
                    netlist_name=netlist_name_base,
                    size_map=size_map,
                    mode="mass_collection" if use_mass_collection else "test_drive"
                )
                
                # re-initialize TuRBO agent for persona
                print(f" {Style.CHECK} Initializing TuRBO-M Agent (Dim: {opt_dim}, M: {run_config['num_trust_regions']})")
                turbo_agent = TurboMSizingGenerator(
                    dim=opt_dim,
                    specs_weights=objective_weights,
                    num_trust_regions=run_config['num_trust_regions'],
                    max_evals=run_config['n_max_evals'],
                    batch_size=run_config['turbo_batch_size'],
                    verbose=True
                )
                
                # reset completion counter for this persona
                total_completed = 0
                turbo_batch_size = run_config['turbo_batch_size']
                
                # use a persona-specific state file so they don't overwrite each other
                turbo_state_persona = os.path.join(turbo_dir, f"turbo_state_{p_name.lower()}.pt")
                
                if os.path.exists(turbo_state_persona):
                    try:
                        turbo_agent.load_state(torch.load(turbo_state_persona))
                        total_completed = len(turbo_agent.X)
                        print_info(f"Resumed TuRBO-M state from {turbo_state_persona} (Completed: {total_completed})")
                    except Exception as e:
                        print_error(f"Failed to load TuRBO-M state: {e}")
                
                # warm starting from Sobol data if in sight mode and no existing TuRBO data
                if len(turbo_agent.X) == 0 and run_config.get('turbo_mode') == 'sight':
                    sobol_parquets = _find_sobol_parquets(netlist_name_base, results_dir)
                    if not sobol_parquets:
                        print_info("Sight mode: No Sobol parquet files found in results directory.")
                    for sobol_pq_path in sobol_parquets:
                        try:
                            print_info(f"Loading Sobol data for warm-start: {sobol_pq_path}")
                            sobol_df = pd.read_parquet(sobol_pq_path)
                            if 'valid' in sobol_df.columns:
                                valid_df = sobol_df[sobol_df['valid'] == True].copy()
                            else:
                                valid_df = sobol_df.copy()
                            
                            if len(valid_df) == 0:
                                print_info("No valid designs in Sobol data, skipping warm-start.")
                            else:
                                X_init, valid_idx = generator.inverse_map(valid_df)
                                
                                if len(X_init) > 0:
                                    valid_df_aligned = valid_df.iloc[valid_idx].reset_index(drop=True)
                                    specs_list = _parquet_to_specs(valid_df_aligned)
                                    
                                    Y_init = turbo_agent.scalarize_specs(specs_list, update_stats=True)
                                    
                                    turbo_agent.load_state(X_init, Y_init)
                                    print_success(f"Warm-started TuRBO-M ({p_name}) with {len(X_init)} designs from {os.path.basename(sobol_pq_path)} (best cost: {Y_init.min().item():.4f})")
                                else:
                                    print_info(f"inverse_map returned 0 valid rows from {os.path.basename(sobol_pq_path)}.")
                        except Exception as e:
                            import traceback
                            print_error(f"Warm-start failed for {os.path.basename(sobol_pq_path)}: {e}")
                            traceback.print_exc()
                            print_info("Continuing without this file.")
                    if len(turbo_agent.X) > 0:
                        print_success(f"Sight warm-start complete: {len(turbo_agent.X)} total points loaded.")
                        
                while total_completed < n_max_evals:
                    curr_batch_size = min(turbo_batch_size, n_max_evals - total_completed)
                    print(f"\n {Style.ARROW} TuRBO Asking for {curr_batch_size} candidates...")
                    context_u = generator.sample_context_u(curr_batch_size)
                    X_next = turbo_agent.ask(
                        curr_batch_size,
                        context_u=context_u,
                        context_dim=len(generator.fixed_params),
                    )
                    X_list = X_next.tolist()
                    batch_samples = generator.generate(curr_batch_size, u_samples=X_list)
                    _validate_sample_context(batch_samples, "turbo batch generation")
                    total_sims = len(batch_samples)
                    for s in batch_samples:
                        s['run_gatekeeper'] = 1
                        s['run_full_char'] = 1
                    print(f" {Style.ARROW} Simulating Contextual Batch ({total_sims} sims for {curr_batch_size} candidates)...")
                    sim_results = []
                    for (b_completed, b_total, b_elapsed, data) in run_parallel_simulations(batch_samples, extractor, n_workers, adaptive=adaptive_workers):
                        rate = b_completed / b_elapsed if b_elapsed > 0 else 0
                        print(f"   Batch Progress: {b_completed}/{total_sims} | Rate: {rate:.1f} sim/s", end='\r')
                        if data:
                            idx, result_val = data
                            sim_results.append(data)
                            if use_mass_collection and len(result_val) == 3:
                                full_res = result_val[2]
                                flat_config = batch_samples[idx]
                                collector.log(
                                    flat_config,
                                    full_res['specs'],
                                    meta={
                                        'sim_id': full_res['id'],
                                        'sim_status': full_res.get('sim_status', -1),
                                        'algorithm': f"turbo_m_{p_name}",
                                    }
                                )
                    results_map = {}
                    for idx, val_tuple in sim_results:
                        if len(val_tuple) >= 2 and isinstance(val_tuple[1], dict):
                            results_map[idx] = val_tuple[:2]
                    
                    completed_specs = []
                    completed_indices = []
                    for idx in range(curr_batch_size):
                        result = results_map.get(idx)
                        if result:
                            completed_specs.append(result[1])
                            completed_indices.append(idx)

                    if len(completed_indices) > 0:
                        X_completed = X_next[completed_indices]
                        turbo_agent.tell(X_completed, completed_specs)
                        total_completed += len(completed_indices)
                        best_v = min([s.best_value for s in turbo_agent.state]) if turbo_agent.state else 0.0
                        print(f"\n {Style.CHECK} Contextual Batch processed. Total: {total_completed}/{n_max_evals} Best Cost: {best_v:.4f}")
                    # save TuRBO-M state after each batch
                    try:
                        torch.save({
                            'state': turbo_agent.state,
                            'X': turbo_agent.X,
                            'Y': turbo_agent.Y,
                            'spec_stats': turbo_agent.spec_stats,
                            'weights': turbo_agent.weights
                        }, turbo_state_persona)
                    except Exception as e:
                        print_error(f"Failed to save TuRBO-M state: {e}")

    except KeyboardInterrupt:
        print(f"\n\n {Style.X} Simulation interrupted for {netlist_name_base}.")
        interrupted = True
        
    finally:
        if collector:
            print(f" {Style.INFO} Finalizing Data Collector...")
            try:
                if interrupted:
                    try:
                        collector.flush()
                    except Exception:
                        pass
                    collector.finalize(discard_partial=False)
                else:
                    collector.finalize(discard_partial=False)
            except Exception as e:
                print_error(f"DataCollector finalize failed: {e}")
        print(f"\n\n {Style.CHECK} Pipeline finished for this netlist.")


# ===== Main ===== #


def main():
    """
    Main entry point for the CLI application.
    """
    clear_screen()
    print_header()

    # 1. netlist selection
    netlist_queue = get_netlist_selection()

    # 2. parallel core configs
    max_cores = mp.cpu_count()
    print_section("Parallel Compute Configuration")
    print(f" {Style.INFO} System has {max_cores} CPU cores available.")
    print(f" {Style.INFO} Worker count is ADAPTIVE — adjusts automatically based on real-time load.")
    
    try:
        load1, _, _ = os.getloadavg()
        print(f" {Style.INFO} Current 1-min CPU load average: {load1:.2f}")
        initial_workers = max(1, int(max_cores - load1) - 2)
        print(f" {Style.INFO} Initial worker estimate: {initial_workers}")
    except AttributeError:
        initial_workers = max(1, max_cores - 2)
    
    print(f"\n Enter 0 for fully adaptive (recommended), or a fixed number.")
    n_workers = get_valid_int(
        "Number of Workers (0 = adaptive)",
        min_val=0,
        max_val=max_cores,
        default=0
    )
    
    adaptive_workers = (n_workers == 0)
    if adaptive_workers:
        n_workers = initial_workers
        print_success(f"Adaptive mode enabled. Starting with {n_workers} workers.")
    else:
        print_success(f"Fixed mode: {n_workers} workers.")

    # 3. algorithm selection
    print_section("Algorithm Selection")
    print(" Select Optimization Algorithm:")
    print("   1. Sobol Sequence (Design Space Exploration)")
    print("   2. TuRBO-M (Integrative Bayesian Optimization)")
    print()
    
    algo_selection = get_valid_int("Enter selection", 1, 2, 1)
    use_turbo = (algo_selection == 2)

    turbo_batch_size = 64
    n_max_evals = 1000
    num_trust_regions = 10
    
    # default configs, will be overridden by user
    run_config = {
        "n_workers": n_workers,
        "adaptive_workers": adaptive_workers,
        "use_turbo": use_turbo,
        "n_max_evals": 1000,
        "turbo_batch_size": 64,
        "num_trust_regions": num_trust_regions,
        "use_mass_collection": False
    }

    if use_turbo:
        print(f" {Style.CHECK} Selected TuRBO-M")
        
        # persona selection
        print("\n Select Optimization Persona (Defines Primary/Secondary Goals):")
        print("   1. Speed (High UGBW/Slew, strict PM, Power limit)")
        print("   2. Precision (High Gain/CMRR/PSRR, Low Offset/THD)")
        print("   3. Efficiency (Low Power/Noise, min UGBW)")
        print("   4. Compactness (Min Estimated Area, good Swing/Slew)")
        print("   5. Balanced (General Purpose - Demo Default)")
        print("   6. Robustness (High PM/CMRR/PSRR, lower sensitivity)")
        print("   7. Linearity (Low THD, strong swing/fidelity)")
        print("   8. Low Headroom (Operate well under constrained VDD)")
        print("   9. Startup Reliability (Fast settling, stable operating point)")
        print("  10. Drive Load (High slew/swing under heavier loading)")
        print("  11. ALL (Pareto Sweep - Runs 1-10 sequentially for Mass Data Collection)")
        
        persona = get_valid_int("Enter persona", 1, 11, 11)
        
        # persona dictionary mapping
        personas_to_run = []
        if persona == 11:
            print(f" {Style.INFO} Persona: PARETO SWEEP (All 10 Personas)")
            personas_to_run = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            personas_to_run = [persona]
            
        run_config['personas_to_run'] = personas_to_run
        
        turbo_batch_size = get_valid_int(
            "Batch Size (Number of candidates per iteration)",
            min_val=1,
            max_val=1000,
            default=64
        )
        n_max_evals = get_valid_int(
            "Total Maximum Evaluations",
            min_val=turbo_batch_size,
            max_val=10000,
            default=640
        )
        
        run_config['turbo_batch_size'] = turbo_batch_size
        run_config['n_max_evals'] = n_max_evals
        
        # determine if injecting preexisting Sobol samples
        run_config['turbo_mode'] = get_turbo_mode(use_turbo)
        
    else:
        print(f" {Style.CHECK} Selected Sobol Explorer")
        n_samples = get_valid_int(
            "Number of Samples to Generate (Per Netlist)",
            min_val=1,
            max_val=1000000,
            default=100
        )
        n_max_evals = n_samples
        run_config['n_max_evals'] = n_max_evals


    # 4. simulation mode
    run_config['use_mass_collection'] = True
    
    print_section("Simulation Mode")
    print(" Only Mass Collection mode is supported. Data will be saved in batch format.")
    sim_mode = "complete"
    run_config['sim_mode'] = sim_mode
    
    # 5. establish output directory
    print_section("Output Directory")
    while True:
        out_dir = input(f" {Style.ARROW} Enter relative path from aspector_core to save results (e.g., results_mtlcad): ").strip()
        if out_dir:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            out_dir = os.path.join(project_root, out_dir)
            print_success(f"Results will be saved to: {out_dir}")
            run_config['output_dir'] = out_dir
            break
        else:
            print_error("You must specify an output directory.")
    
    print()
    print(Style.DOUBLE_LINE)
    print(f"      STARTING BATCH JOB | {len(netlist_queue)} NETLISTS       ".center(70))
    print(Style.DOUBLE_LINE)
    print()
    
    # 6. batch loop
    for i, (name, path) in enumerate(netlist_queue):
        print(f"\n[{i+1}/{len(netlist_queue)}] Running Task: {name}")
        run_optimization_task(name, path, run_config)
    
    print(f"\n\n {Style.CHECK} All Tasks Completed.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
