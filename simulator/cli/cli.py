"""
cli.py

Author: natelgrw
Last Edited: 01/24/2026

Command Line Interface for ASPECTOR Core.
Provides interactive setup for running circuit optimization pipelines.
Supports Parallel Execution.
"""

import os
import sys
import argparse
import numpy as np
import time
import multiprocessing as mp
import pandas as pd
# from functools import partial # Removed as part of cleanup

# Add workspace to path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from simulator import globalsy
from simulator.compute.runner import run_parallel_simulations
from simulator.compute.collector import DataCollector
from algorithms.sobol.generator import SobolSizingGenerator
from algorithms.turbo_m import ASPECTOR_TurboM
from simulator.eval_engines.utils.design_reps import extract_sizing_map
from simulator.eval_engines.spectre.configs.config_env import EnvironmentConfig
from simulator.eval_engines.extractor.extractor import Extractor, extract_parameter_names, classify_opamp_type
import torch

# ASCII Art and Styles
class Style:
    # Use simple unicode/ascii characters instead of colors
    CHECK = "[+]"
    X = "[!]"
    INFO = "..."
    ARROW = ">>"
    LINE = "-" * 70
    DOUBLE_LINE = "=" * 70

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print()
    print(Style.DOUBLE_LINE)
    print("       TITAN FOUNDATION MODEL - CIRCUIT OPTIMIZATION PIPELINE       ".center(70))
    print(Style.DOUBLE_LINE)
    print()

def print_section(title):
    print(f"\n{Style.LINE}")
    print(f" {title.upper()} ".center(70))
    print(f"{Style.LINE}\n")

def print_success(message):
    print(f" {Style.CHECK} {message}")

def print_info(message):
    print(f" {Style.INFO} {message}")

def print_error(message):
    print(f" {Style.X} Error: {message}")

def get_valid_int(prompt, min_val, max_val, default):
    """
    Prompt for an integer within a range.
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
    default_path = os.path.join(project_root, "demo_netlists")
    
    print(f" Default Directory: {default_path}")
    print(" (Press ENTER to use default, or paste a custom folder path)")
    
    custom_path = input(f" {Style.ARROW} Directory: ").strip()
    
    base_path = default_path
    if custom_path:
        # Handle relative paths, user paths (~), etc.
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
            return [] # Should probably loop or exit, but let's handle graceful loop or fallback
            
        print("\n Available Netlists:")
        for i, f in enumerate(available_files):
            print(f"   {i+1}. {f[:-4]}")
        print(f"   {len(available_files)+1}. [BATCH] Run ALL Netlists in folder")
        print()
    else:
        print_error(f"Path does not exist: {base_path}")
        return get_netlist_selection() # Recursive retry

    while True:
        prompt = f" {Style.ARROW} Enter selection (Number or Name): "
        user_input = input(prompt).strip()
        
        if not user_input:
            continue
            
        # Check for "ALL" / Number Selection
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(available_files):
                # Single File
                fname = available_files[idx]
                return [(fname[:-4], os.path.join(base_path, fname))]
            elif idx == len(available_files):
                # ALL
                print_info(f"Selected Batch Mode: {len(available_files)} netlists")
                return [(f[:-4], os.path.join(base_path, f)) for f in available_files]
            else:
                 print_error("Invalid selection number")
                 continue
        
        # Fallback to name string match
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
    Prompts for Blind vs Sight (Warm Start) mode.
    Returns: (mode_string, path_string_or_None)
    """
    if not use_turbo:
        return "blind", None

    print("\n TuRBO Initialization Mode:")
    print("   1. Blind (Default) - Start from scratch (Latin Hypercube / Sobol)")
    print("   2. Sight (Warm Start) - Load existing Parquet data")
    
    while True:
        try:
            sel = input(f" {Style.ARROW} Enter selection [1]: ").strip()
            if not sel:
                sel = "1"
            
            if sel == "1":
                return "blind", None
            elif sel == "2":
                print_info("Enter path to Parquet file (Single Mode) or Directory (Batch Mode)")
                print("     For Batch Mode: Directory must contain subfolders matching netlist names.")
                path = input(f" {Style.ARROW} Path: ").strip()
                path = os.path.expanduser(path)
                
                if os.path.exists(path):
                    print_success(f"Path verified: {path}")
                    return "sight", os.path.abspath(path)
                else:
                    print_error("Path does not exist.")
                    continue
            else:
                print_error("Invalid selection.")
        except ValueError:
            pass

def run_optimization_task(netlist_name_base, scs_file_path, run_config):
    """
    Executes the optimization pipeline for a single netlist.
    """
    print_section(f"Processing: {netlist_name_base}")
    
    # Unpack Config
    n_workers = run_config['n_workers']
    use_turbo = run_config['use_turbo']
    use_mass_collection = run_config['use_mass_collection']
    

    # 5. Pipeline Execution Setup (Relative to netlist)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    results_dir = os.path.join(project_root, "results", netlist_name_base)

    # Subdirectories for sobol, turbo_m
    sobol_dir = os.path.join(results_dir, "sobol")
    turbo_dir = os.path.join(results_dir, "turbo_m")

    # Ensure directories exist
    os.makedirs(sobol_dir, exist_ok=True)
    os.makedirs(turbo_dir, exist_ok=True)

    # DataCollector and state file paths
    collector = None
    sobol_parquet = os.path.join(sobol_dir, f"{netlist_name_base}_sobol.parquet")
    sobol_state = os.path.join(sobol_dir, "sobol_state.txt")
    turbo_parquet = os.path.join(turbo_dir, f"{netlist_name_base}_turbo_m.parquet")
    turbo_state = os.path.join(turbo_dir, "turbo_state.pt")

    # Choose collector output dir based on algorithm
    if use_mass_collection:
        if use_turbo:
            collector = DataCollector(output_dir=turbo_dir, buffer_size=1000, parquet_name=f"{netlist_name_base}_turbo_m.parquet")
        else:
            collector = DataCollector(output_dir=sobol_dir, buffer_size=1000, parquet_name=f"{netlist_name_base}_sobol.parquet")

    size_map = extract_sizing_map(scs_file_path)

    # Extract parameters
    params_id = extract_parameter_names(scs_file_path)
    ignored_params = ['fet_num', 'vdd', 'vcm', 'tempc', 'rfeedback_val', 'rsrc_val', 'cload_val']
    sizing_params_for_gen = [p for p in params_id if p not in ignored_params]
    
    # Initialize Generator (Mapping Layer)
    generator = SobolSizingGenerator(sizing_params_for_gen, seed=None)
    
    # Define Optimization Dimension (Sizing only, Env handled stochastically)
    opt_dim = generator.dim_sizing if use_turbo else generator.dim

    # Configure Extractor
    sim_flags = {'ac': True, 'dc': True, 'noise': True, 'tran': True}
    opamp_type = classify_opamp_type(scs_file_path)
    
    full_lb = [-1e9] * len(params_id)
    full_ub = [ 1e9] * len(params_id)
    
    config_env = EnvironmentConfig(scs_file_path, opamp_type, {}, params_id, full_lb, full_ub, results_dir=turbo_dir if use_turbo else sobol_dir)
    yaml_path = config_env.write_yaml_configs()
    
    # Default Optimization Goals for Demo (Maximize Gain/BW, Minimal Power)
    # If running multiple personas, we will re-initialize the extractor and agent inside the loop
    if not use_turbo:
        specs_id = ["gain_ol", "ugbw", "pm", "power", "vos"]
        specs_ideal = [0.0] * len(specs_id) 
        specs_weights = [1.0, 1.0, 10.0, 10.0, 10.0]
        
        extractor = Extractor(
            dim=len(params_id),
            opt_params=params_id,
            params_id=params_id,
            specs_id=specs_id,            
            specs_ideal=specs_ideal,
            specs_weights=specs_weights,
            sim_flags=sim_flags,
            vcm=0,                 
            vdd=0,                 
            tempc=27,              
            ub=full_ub,            
            lb=full_lb,            
            yaml_path=yaml_path,
            fet_num=0,             
            results_dir=sobol_dir, 
            netlist_name=netlist_name_base,
            size_map=size_map,
            mode="mass_collection" if use_mass_collection else "test_drive",
            sim_mode=run_config.get('sim_mode', 'complete')
        )

    # Shared Execution Loop Variables
    n_max_evals = run_config['n_max_evals']
    

    try:
        if not use_turbo:
            # --- SOBOL MODE (One-shot batch) ---
            print(f" {Style.INFO} Mode: Sobol Exploration (One-shot)")
            # Resume logic: check for sobol_state.txt
            start_idx = 0
            if os.path.exists(sobol_state):
                try:
                    with open(sobol_state, 'r') as f:
                        start_idx = int(f.read().strip())
                    print_info(f"Resuming Sobol sequence from index {start_idx}")
                except Exception as e:
                    print_error(f"Failed to read Sobol state: {e}")
            samples = generator.generate(n_max_evals, start_idx=start_idx)
            # Inject Flags
            for s in samples:
                s['run_gatekeeper'] = 1
                s['run_full_char'] = 1
            completed = 0
            for (completed, total, elapsed, data) in run_parallel_simulations(samples, extractor, n_workers):
                rate = completed / elapsed if elapsed > 0 else 0
                percent = (completed / total) * 100
                bar = '#' * int(30 * completed // total) + '-' * (30 - int(30 * completed // total))
                print(f" [{bar}] {percent:5.1f}% | {completed}/{total} | Rate: {rate:4.1f} sim/s", end='\r')
                if data and use_mass_collection and collector:
                    idx, result_val = data
                    if len(result_val) == 3:
                        full_res = result_val[2]
                        flat_config = samples[idx]
                        collector.log(
                            flat_config,
                            full_res['specs'],
                            meta={
                                'sim_id': full_res['id'],
                                'algorithm': 'sobol',
                                'netlist_name': netlist_name_base
                            }
                        )
                
                # Periodically save Sobol state to survive hard crashes
                if completed % 1000 == 0:
                    try:
                        with open(sobol_state, 'w') as f:
                            f.write(str(start_idx + completed))
                    except Exception:
                        pass

            # Save final state
            try:
                with open(sobol_state, 'w') as f:
                    f.write(str(start_idx + n_max_evals))
                print_info(f"Saved Sobol state at index {start_idx + n_max_evals}")
            except Exception as e:
                print_error(f"Failed to save Sobol state: {e}")
        else:
            # --- TURBO MODE (Iterative Loop) ---
            personas_to_run = run_config.get('personas_to_run', [5])
            
            for p_idx in personas_to_run:
                # Set weights based on persona
                if p_idx == 1:
                    selected_weights = {'ugbw': 10.0, 'slew_rate': 10.0, 'settling_time': 10.0, 'pm': 100.0, 'gm': 50.0, 'power': 5.0, 'gain_ol': 1.0}
                    p_name = "SPEED"
                elif p_idx == 2:
                    selected_weights = {'gain_ol': 10.0, 'cmrr': 10.0, 'psrr': 10.0, 'vos': 10.0, 'thd': 10.0, 'pm': 100.0, 'gm': 50.0, 'integrated_noise': 5.0, 'ugbw': 1.0}
                    p_name = "PRECISION"
                elif p_idx == 3:
                    selected_weights = {'power': 20.0, 'integrated_noise': 10.0, 'pm': 100.0, 'gm': 50.0, 'ugbw': 5.0, 'gain_ol': 1.0}
                    p_name = "EFFICIENCY"
                elif p_idx == 4:
                    selected_weights = {'area': 20.0, 'pm': 100.0, 'gm': 50.0, 'output_voltage_swing': 5.0, 'slew_rate': 5.0, 'ugbw': 1.0, 'gain_ol': 1.0}
                    p_name = "COMPACTNESS"
                else:
                    selected_weights = {'gain_ol': 1.0, 'ugbw': 1.0, 'pm': 100.0, 'gm': 50.0, 'power': 2.0, 'vos': 5.0}
                    p_name = "BALANCED"
                    
                print(f"\n {Style.DOUBLE_LINE}")
                print(f" {Style.INFO} Starting TuRBO-M Optimization Loop - Persona: {p_name}")
                print(f" {Style.DOUBLE_LINE}")
                
                # Re-initialize Extractor for this persona
                specs_id = list(selected_weights.keys())
                specs_ideal = [0.0] * len(specs_id) 
                specs_weights_list = list(selected_weights.values())
                
                extractor = Extractor(
                    dim=len(params_id),
                    opt_params=params_id,
                    params_id=params_id,
                    specs_id=specs_id,            
                    specs_ideal=specs_ideal,
                    specs_weights=specs_weights_list,
                    sim_flags=sim_flags,
                    vcm=0,                 
                    vdd=0,                 
                    tempc=27,              
                    ub=full_ub,            
                    lb=full_lb,            
                    yaml_path=yaml_path,
                    fet_num=0,             
                    results_dir=turbo_dir, 
                    netlist_name=netlist_name_base,
                    size_map=size_map,
                    mode="mass_collection" if use_mass_collection else "test_drive",
                    sim_mode=run_config.get('sim_mode', 'complete')
                )
                
                # Re-initialize TuRBO Agent for this persona
                print(f" {Style.CHECK} Initializing TuRBO-M Agent (Dim: {opt_dim}, M: {run_config['num_trust_regions']})")
                turbo_agent = ASPECTOR_TurboM(
                    dim=opt_dim,
                    specs_weights=selected_weights,
                    num_trust_regions=run_config['num_trust_regions'],
                    max_evals=run_config['n_max_evals'],
                    batch_size=run_config['turbo_batch_size'],
                    verbose=True
                )
                
                # Reset completion counter for this persona
                total_completed = 0
                turbo_batch_size = run_config['turbo_batch_size']
                
                # We use a persona-specific state file so they don't overwrite each other
                turbo_state_persona = os.path.join(turbo_dir, f"turbo_state_{p_name.lower()}.pt")
                
                # Resume logic: check for turbo_state.pt
                if os.path.exists(turbo_state_persona):
                    try:
                        turbo_agent.load_state(torch.load(turbo_state_persona))
                        # Calculate how many we've already done based on the state
                        total_completed = len(turbo_agent.X) // 2 # Divide by 2 because of robust pairs
                        print_info(f"Resumed TuRBO-M state from {turbo_state_persona} (Completed: {total_completed})")
                    except Exception as e:
                        print_error(f"Failed to load TuRBO-M state: {e}")
                        
                while total_completed < n_max_evals:
                    curr_batch_size = min(turbo_batch_size, n_max_evals - total_completed)
                    print(f"\n {Style.ARROW} TuRBO Asking for {curr_batch_size} candidates...")
                    X_next = turbo_agent.ask(curr_batch_size)
                    X_list = X_next.tolist()
                    samples_nom = generator.generate(curr_batch_size, u_samples=X_list, robust_env=False)
                    samples_rob = generator.generate(curr_batch_size, u_samples=X_list, robust_env=True)
                    full_batch_samples = samples_nom + samples_rob
                    total_sims = len(full_batch_samples)
                    for s in full_batch_samples:
                        s['run_gatekeeper'] = 1
                        s['run_full_char'] = 1
                    print(f" {Style.ARROW} Simulating Robust Batch ({total_sims} sims for {curr_batch_size} candidates)...")
                    sim_results = []
                    for (b_completed, b_total, b_elapsed, data) in run_parallel_simulations(full_batch_samples, extractor, n_workers):
                        rate = b_completed / b_elapsed if b_elapsed > 0 else 0
                        print(f"   Batch Progress: {b_completed}/{total_sims} | Rate: {rate:.1f} sim/s", end='\r')
                        if data:
                            idx, result_val = data
                            sim_results.append(data)
                            if use_mass_collection and len(result_val) == 3:
                                full_res = result_val[2]
                                flat_config = full_batch_samples[idx]
                                collector.log(
                                    flat_config,
                                    full_res['specs'],
                                    meta={
                                        'sim_id': full_res['id'],
                                        'algorithm': 'turbo_m',
                                        'persona': p_name,
                                        'netlist_name': netlist_name_base
                                    }
                                )
                    
                    # Filter valid results only for TuRBO update
                    results_map = {}
                    for idx, val_tuple in sim_results:
                        # Check if tuple has at least reward and specs (length >= 2) and specs is dict
                        # val_tuple structure: (reward, specs) or (reward, specs, full_res) or (err_msg)
                        if len(val_tuple) >= 2 and isinstance(val_tuple[1], dict):
                            results_map[idx] = val_tuple[:2]
                    
                    ordered_specs = []
                    valid_indices = []
                    n_candidates = curr_batch_size
                    for i in range(n_candidates):
                        idx_nom = i
                        idx_rob = i + n_candidates
                        res_nom = results_map.get(idx_nom)
                        res_rob = results_map.get(idx_rob)
                        if res_nom and res_rob:
                            specs_nom = res_nom[1]
                            specs_rob = res_rob[1]
                            worst_case_specs = {}
                            all_keys = set(specs_nom.keys()) | set(specs_rob.keys())
                            worst_case_specs['valid'] = specs_nom.get('valid', True) and specs_rob.get('valid', True)
                            for key in all_keys:
                                if key == 'valid': continue
                                val_n = specs_nom.get(key)
                                val_r = specs_rob.get(key)
                                if val_n is None or val_r is None: continue
                                def is_better(v1, v2, k):
                                    if k in ["power", "integrated_noise", "settling_time", "vos", "thd", "ibias"]:
                                        return abs(v1) < abs(v2)
                                    else:
                                        return v1 > v2
                                if is_better(val_n, val_r, key):
                                    worst_case_specs[key] = val_r
                                else:
                                    worst_case_specs[key] = val_n
                            ordered_specs.append(worst_case_specs)
                            valid_indices.append(i)
                    if len(valid_indices) > 0:
                        X_valid = X_next[valid_indices]
                        turbo_agent.tell(X_valid, ordered_specs)
                        total_completed += len(valid_indices)
                        best_v = min([s.best_value for s in turbo_agent.state]) if turbo_agent.state else 0.0
                        print(f"\n {Style.CHECK} Robust Batch (Worst-Case) processed. Total: {total_completed}/{n_max_evals} Best Cost: {best_v:.4f}")
                    # Save TuRBO-M state after each batch
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
        
    finally:
        config_env.del_yaml()
        if collector:
            print(f" {Style.INFO} Finalizing Data Collector...")
            collector.finalize()
        print(f"\n\n {Style.CHECK} Pipeline finished for this netlist.")


def main():
    clear_screen()
    print_header()

    # 1. Netlist Selection (Returns List)
    netlist_queue = get_netlist_selection()

    # 2. Parallel Core Configuration
    max_cores = mp.cpu_count()
    print_section("Parallel Compute Configuration")
    print(f" {Style.INFO} System has {max_cores} CPU cores available.")
    
    n_workers = get_valid_int(
        "Number of Parallel Workers (Simulations to run at once)",
        min_val=1,
        max_val=max_cores,
        default=max(1, max_cores - 2)
    )

    # 3. Algorithm Selection
    print_section("Algorithm Selection")
    print(" Select Optimization Algorithm:")
    print("   1. Sobol Sequence (Design Space Exploration)")
    print("   2. TuRBO-M (Integrative Bayesian Optimization)")
    print()
    
    algo_selection = get_valid_int("Enter selection", 1, 2, 1)
    use_turbo = (algo_selection == 2)

    # Algorithm specific settings (shared across all netlists)
    turbo_batch_size = 64
    n_max_evals = 1000
    num_trust_regions = 10  # Increased for mass data collection
    
    selected_weights = {} 
    
    # Defaults
    run_config = {
        "n_workers": n_workers,
        "use_turbo": use_turbo,
        "n_max_evals": 1000,
        "turbo_batch_size": 64,
        "num_trust_regions": num_trust_regions,
        "selected_weights": {},
        "use_mass_collection": False
    }

    if use_turbo:
        print(f" {Style.CHECK} Selected TuRBO-M")
        
        # --- Persona Selection ---
        print("\n Select Optimization Persona (Defines Primary/Secondary Goals):")
        print("   1. Speed (High UGBW/Slew, strict PM, Power limit)")
        print("   2. Precision (High Gain/CMRR/PSRR, Low Offset/THD)")
        print("   3. Efficiency (Low Power/Noise, min UGBW)")
        print("   4. Compactness (Min Area, good Swing/Slew)")
        print("   5. Balanced (General Purpose - Demo Default)")
        print("   6. ALL (Pareto Sweep - Runs 1-5 sequentially for Mass Data Collection)")
        
        persona = get_valid_int("Enter persona", 1, 6, 6)
        
        # ... Persona Dictionary Mapping ...
        personas_to_run = []
        if persona == 6:
            print(f" {Style.INFO} Persona: PARETO SWEEP (All 5 Personas)")
            personas_to_run = [1, 2, 3, 4, 5]
        else:
            personas_to_run = [persona]
            
        # We will store the selected weights in a list if running multiple
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
        
        # Determine Blind vs Sight Mode
        t_mode, t_path = get_turbo_mode(use_turbo)
        run_config['turbo_mode'] = t_mode
        run_config['turbo_data_path'] = t_path
        
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

    # 4. Data Collection Mode
    print_section("Data Collection Mode")
    print(" Select Data Output Strategy:")
    print("   1. Test Drive (JSON per sim)")
    print("   2. Mass Collection (Parquet, Batch JSONs)")
    
    data_mode_sel = get_valid_int("Enter selection", 1, 2, 1)
    use_mass_collection = (data_mode_sel == 2)
    run_config['use_mass_collection'] = use_mass_collection
    
    # 5. Simulation Mode
    print_section("Simulation Mode")
    print(" Select Simulation Mode:")
    print("   1. Complete Mode (Runs all simulations regardless of DC operating point)")
    print("   2. Efficient Mode (Runs DC first, only runs AC/Tran if transistors are in valid operating regions)")
    
    sim_mode_sel = get_valid_int("Enter selection", 1, 2, 1)
    sim_mode = "complete" if sim_mode_sel == 1 else "efficient"
    run_config['sim_mode'] = sim_mode
    
    print()
    print(Style.DOUBLE_LINE)
    print(f"      STARTING BATCH JOB | {len(netlist_queue)} NETLISTS       ".center(70))
    print(Style.DOUBLE_LINE)
    print()
    
    # START BATCH LOOP
    for i, (name, path) in enumerate(netlist_queue):
        print(f"\n[{i+1}/{len(netlist_queue)}] Running Task: {name}")
        run_optimization_task(name, path, run_config)
    
    print(f"\n\n {Style.CHECK} All Tasks Completed.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
