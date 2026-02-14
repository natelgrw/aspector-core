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
from simulator.eval_engines.utils.netlist_to_graph import parse_netlist_to_graph, extract_sizing_map
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

    # Subdirectories for graph, sobol, turbo_m
    graph_dir = os.path.join(results_dir, "graph")
    sobol_dir = os.path.join(results_dir, "sobol")
    turbo_dir = os.path.join(results_dir, "turbo_m")

    # Ensure directories exist
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(sobol_dir, exist_ok=True)
    os.makedirs(turbo_dir, exist_ok=True)

    # DataCollector and state file paths
    collector = None
    sobol_parquet = os.path.join(sobol_dir, f"{netlist_name_base}_sobol.parquet")
    sobol_state = os.path.join(sobol_dir, "sobol_state.txt")
    turbo_parquet = os.path.join(turbo_dir, f"{netlist_name_base}_turbo_m.parquet")
    turbo_state = os.path.join(turbo_dir, "turbo_state.pt")

    sims_dir = os.path.join(results_dir, "simulations")
    if not os.path.exists(sims_dir):
        os.makedirs(sims_dir, exist_ok=True)

    # Choose collector output dir based on algorithm
    if use_mass_collection:
        if use_turbo:
            collector = DataCollector(output_dir=turbo_dir, buffer_size=50, parquet_name=f"{netlist_name_base}_turbo_m.parquet")
        else:
            collector = DataCollector(output_dir=sobol_dir, buffer_size=50, parquet_name=f"{netlist_name_base}_sobol.parquet")

    print(f" {Style.CHECK} Generating Graph Representation")
    graph_fmt = "pt" if use_mass_collection else "json"
    parse_netlist_to_graph(scs_file_path, graph_dir, sim_id=1, topology_id=netlist_name_base, format=graph_fmt)
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
    
    config_env = EnvironmentConfig(scs_file_path, opamp_type, {}, params_id, full_lb, full_ub)
    yaml_path = config_env.write_yaml_configs()
    
    # Default Optimization Goals for Demo (Maximize Gain/BW, Minimal Power)
    specs_id = list(run_config['selected_weights'].keys()) if use_turbo else ["gain", "ugbw", "pm", "power", "vos"]
    specs_ideal = [0.0] * len(specs_id) 
    specs_weights = list(run_config['selected_weights'].values()) if use_turbo else [1.0, 1.0, 10.0, 10.0, 10.0]
    
    extractor = Extractor(
        dim=len(params_id),
        opt_params=params_id,
        params_id=params_id,
        specs_id=specs_id if use_turbo else [],            
        specs_ideal=specs_ideal if use_turbo else [],
        specs_weights=specs_weights if use_turbo else [],
        sim_flags=sim_flags,
        vcm=0,                 
        vdd=0,                 
        tempc=27,              
        ub=full_ub,            
        lb=full_lb,            
        yaml_path=yaml_path,
        fet_num=0,             
        results_dir=results_dir, 
        netlist_name=netlist_name_base,
        size_map=size_map,
        mode="mass_collection" if use_mass_collection else "test_drive"
    )

    # Initialize TuRBO if selected
    turbo_agent = None
    if use_turbo:
        print(f" {Style.CHECK} Initializing TuRBO-M Agent (Dim: {opt_dim})")
        turbo_agent = ASPECTOR_TurboM(
            dim=opt_dim,
            specs_weights=run_config['selected_weights'],
            max_evals=run_config['n_max_evals'],
            batch_size=run_config['turbo_batch_size'],
            verbose=True
        )
        
        # --- Handle Sight Mode (Warm Start) ---
        if run_config.get('turbo_mode') == 'sight':
            data_path = run_config.get('turbo_data_path')
            target_parquet = None
            
            # Check if path is file or dir
            if os.path.isfile(data_path):
                target_parquet = data_path
            elif os.path.isdir(data_path):
                # Batch Mode: Look for subdir matching netlist_name_base
                search_dir = os.path.join(data_path, netlist_name_base)
                
                # Verify existence
                if not os.path.exists(search_dir):
                     # Fallback: Maybe the directory IS the netlist directory?
                     # Check if data_path contains our parsable files?
                     pass
                else:
                     # Find parquet recursively
                     parquets = []
                     for root, dirs, files in os.walk(search_dir):
                         for f in files:
                             if f.endswith('.parquet'):
                                 parquets.append(os.path.join(root, f))
                     
                     if parquets:
                         target_parquet = max(parquets, key=os.path.getsize)
                         print_info(f"Found dataset: {os.path.basename(target_parquet)}")
                     else:
                         print_error(f"Sight Mode: No .parquet file found in {search_dir}")
            
            if target_parquet:
                print(f" {Style.INFO} Loading Warm-Start Data from: {target_parquet}")
                try:
                    df = pd.read_parquet(target_parquet)
                    
                    # 1. Inverse Map X
                    print(f"   Mapping {len(df)} points to Unit Hypercube...")
                    X_init = generator.inverse_map(df)
                    
                    # 2. Extract Specs (for Y generation)
                    specs_list = []
                    target_specs = run_config['selected_weights'].keys()
                    
                    for idx, row in df.iterrows():
                        spec_entry = {}
                        spec_entry['valid'] = bool(row.get('valid', True))
                            
                        # Extract metrics
                        all_present = True
                        for k in target_specs:
                            col = f"out_{k}"
                            if col in row:
                                spec_entry[k] = row[col]
                            elif k in row:
                                spec_entry[k] = row[k]
                            else:
                                if spec_entry['valid']: all_present = False
                        
                        # Extra hard constraints
                        if 'out_pm' in row: spec_entry['pm'] = row['out_pm']
                        elif 'pm' in row: spec_entry['pm'] = row['pm']
                        if 'out_gm' in row: spec_entry['gm'] = row['out_gm']
                        elif 'gm' in row: spec_entry['gm'] = row['gm']

                        if all_present:
                            specs_list.append(spec_entry)
                        else:
                            specs_list.append({'valid': False})
                            
                    # 3. Scalarize and Update
                    if len(X_init) > 0:
                        Y_init = turbo_agent.scalarize_specs(specs_list, update_stats=True)
                        turbo_agent.load_state(X_init, Y_init)
                        print(f" {Style.CHECK} Warm-Start Complete. Loaded {len(X_init)} samples.")
                    else:
                        print_error("No valid samples mapped from dataset.")
                    
                except Exception as e:
                    print_error(f"Failed to load Warm-Start data: {e}")


    # Shared Execution Loop Variables
    total_completed = 0
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
                        collector.log(
                            full_res['parameters'],
                            full_res['specs'],
                            meta={
                                'sim_id': full_res['id'],
                                'algorithm': 'sobol',
                                'netlist_name': netlist_name_base
                            }
                        )
            # Save new state
            try:
                with open(sobol_state, 'w') as f:
                    f.write(str(start_idx + n_max_evals))
                print_info(f"Saved Sobol state at index {start_idx + n_max_evals}")
            except Exception as e:
                print_error(f"Failed to save Sobol state: {e}")
        else:
            # --- TURBO MODE (Iterative Loop) ---
            print(f" {Style.INFO} Mode: TuRBO-M Optimization Loop")
            turbo_batch_size = run_config['turbo_batch_size']
            # Resume logic: check for turbo_state.pt
            if os.path.exists(turbo_state):
                try:
                    turbo_agent.load_state(torch.load(turbo_state))
                    print_info(f"Resumed TuRBO-M state from {turbo_state}")
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
                            collector.log(
                                full_res['parameters'],
                                full_res['specs'],
                                meta={
                                    'sim_id': full_res['id'],
                                    'algorithm': 'turbo_m',
                                    'netlist_name': netlist_name_base
                                }
                            )
                results_map = {idx: val_tuple[:2] for idx, val_tuple in sim_results}
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
                    best_v = turbo_agent.state.best_value if turbo_agent.state.best_value is not None else 0.0
                    print(f"\n {Style.CHECK} Robust Batch (Worst-Case) processed. Total: {total_completed}/{n_max_evals} Best Cost: {best_v:.4f}")
                # Save TuRBO-M state after each batch
                try:
                    torch.save({
                        'state': turbo_agent.state,
                        'X': turbo_agent.X,
                        'Y': turbo_agent.Y,
                        'spec_stats': turbo_agent.spec_stats,
                        'weights': turbo_agent.weights
                    }, turbo_state)
                    print_info(f"Saved TuRBO-M state to {turbo_state}")
                except Exception as e:
                    print_error(f"Failed to save TuRBO-M state: {e}")

    except KeyboardInterrupt:
        print(f"\n\n {Style.X} Simulation interrupted for {netlist_name_base}.")
        
    finally:
        config_env.del_yaml() 
        if collector:
            print(f" {Style.INFO} Flushing Data Collector...")
            collector.flush()

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

    # Algorithm specific settings
    turbo_batch_size = 64
    n_max_evals = 1000
    
    selected_weights = {} # Filled based on persona

    if use_turbo:
        print(f" {Style.CHECK} Selected TuRBO-M")
        
        # --- Persona Selection ---
        print("\n Select Optimization Persona (Defines Primary/Secondary Goals):")
        print("   1. Speed (High UGBW/Slew, strict PM, Power limit)")
        print("   2. Precision (High Gain/CMRR/PSRR, Low Offset/THD)")
        print("   3. Efficiency (Low Power/Noise, min UGBW)")
        print("   4. Compactness (Min Area, good Swing/Slew)")
        print("   5. Balanced (General Purpose - Demo Default)")
        
        persona = get_valid_int("Enter persona", 1, 5, 5)
        
        # Default Baseline weights for all: PM/GM are constraints (100.0/50.0 implicit or explicit)
        # We define explicit weights here to override TuRBO defaults
        # Baseline: PM=100 (Constraint), GM=50 (Constraint)
        # We ensure every persona has these.
        
        if persona == 1: # Speed
            print(f" {Style.INFO} Persona: SPEED")
            selected_weights = {
                'ugbw': 10.0, 'slew_rate': 10.0, 'settling_time': 10.0, 
                'pm': 100.0, 'gm': 50.0, 'power': 5.0, # Secondary
                'gain_ol': 1.0 # Tolerable
            }
        elif persona == 2: # Precision
            print(f" {Style.INFO} Persona: PRECISION")
            selected_weights = {
                'gain_ol': 10.0, 'cmrr': 10.0, 'psrr': 10.0, 'vos': 10.0, 'thd': 10.0,
                'pm': 100.0, 'gm': 50.0, 'integrated_noise': 5.0, # Secondary
                'ugbw': 1.0 # Tolerable
            }
        elif persona == 3: # Efficiency
            print(f" {Style.INFO} Persona: EFFICIENCY")
            selected_weights = {
                'power': 20.0, 'integrated_noise': 10.0, 
                'pm': 100.0, 'gm': 50.0, 'ugbw': 5.0, # Secondary Threshold
                'gain_ol': 1.0 # Tolerable
            }
        elif persona == 4: # Compactness
            print(f" {Style.INFO} Persona: COMPACTNESS")
            selected_weights = {
                'area': 20.0, 
                'pm': 100.0, 'gm': 50.0, 
                'output_voltage_swing': 5.0, 'slew_rate': 5.0, # Secondary
                'ugbw': 1.0, 'gain_ol': 1.0 # Tolerable
            }
        else: # Balanced
            print(f" {Style.INFO} Persona: BALANCED (Default)")
            selected_weights = {
                 'gain_ol': 1.0, 'ugbw': 1.0, 'pm': 100.0, 'gm': 50.0,
                 'power': 2.0, 'vos': 5.0
            }

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
    else:
        print(f" {Style.CHECK} Selected Sobol Explorer")
        n_samples = get_valid_int(
            "Number of Samples to Generate",
            min_val=1,
            max_val=1000000,
            default=100
        )
        n_max_evals = n_samples # For compatibility in loop logic

    # 4. Data Collection Mode
    print_section("Data Collection Mode")
    print(" Select Data Output Strategy:")
    print("   1. Test Drive (Default)")
    print("      - Writes individual JSON files to disk for every simulation.")
    print("      - Good for debugging and small batches (<100 sims).")
    print("      - Slow for large scale data (I/O bottleneck).")
    print("   2. Mass Collection (NeurIPS Standard)")
    print("      - Buffers results in memory.")
    print("      - Appends to high-performance Parquet file every 50 sims.")
    print("      - Saves PyTorch Geometric Graphs (.pt) for ML training.")
    print("      - Minimal I/O overhead.")
    
    data_mode_sel = get_valid_int("Enter selection", 1, 2, 1)
    use_mass_collection = (data_mode_sel == 2)
    
    # Initialize Collector if needed
    collector = None
    if use_mass_collection:
        print_info("Initializing Mass Data Collector...")
        # We will set output path later after project_root is defined
    else:
        print_info("Using Standard I/O (One file per sim)")

    # 5. Pipeline Execution Setup
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    results_dir = os.path.join(project_root, "results", netlist_name_base)
    
    if use_mass_collection:
        collector = DataCollector(output_dir=os.path.join(results_dir, "dataset"), buffer_size=50)

    sims_dir = os.path.join(results_dir, "simulations")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        os.makedirs(os.path.join(results_dir, "graph"))
        if not use_mass_collection:
            os.makedirs(sims_dir) 
    else:
        os.makedirs(os.path.join(results_dir, "graph"), exist_ok=True)
        if not use_mass_collection:
            os.makedirs(sims_dir, exist_ok=True)

    print_section("Initialization")
    
    print(f" {Style.CHECK} Generating Graph Representation")
    parse_netlist_to_graph(scs_file_path, os.path.join(results_dir, "graph"), sim_id=1, topology_id=netlist_name_base)
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
    
    config_env = EnvironmentConfig(scs_file_path, opamp_type, {}, params_id, full_lb, full_ub)
    yaml_path = config_env.write_yaml_configs()
    
    # Default Optimization Goals for Demo (Maximize Gain/BW, Minimal Power)
    # Note: Extractor legacy reward uses LISTS. TuRBO uses DICT (selected_weights).
    # We will pass dummy lists to Extractor if TuRBO is main, just to allow it to initialize.
    # Extractor.reward() is not used by TuRBO loop.
    
    specs_id = list(selected_weights.keys()) if use_turbo else ["gain", "ugbw", "pm", "power", "vos"]
    specs_ideal = [0.0] * len(specs_id) # Dummy, not used by TuRBO
    specs_weights = list(selected_weights.values()) if use_turbo else [1.0, 1.0, 10.0, 10.0, 10.0]
    
    extractor = Extractor(
        dim=len(params_id),
        opt_params=params_id,
        params_id=params_id,
        specs_id=specs_id if use_turbo else [],            
        specs_ideal=specs_ideal if use_turbo else [],
        specs_weights=specs_weights if use_turbo else [],
        sim_flags=sim_flags,
        vcm=0,                 
        vdd=0,                 
        tempc=27,              
        ub=full_ub,            
        lb=full_lb,            
        yaml_path=yaml_path,
        fet_num=0,             
        results_dir=results_dir, 
        netlist_name=netlist_name_base,
        size_map=size_map,
        mode="mass_collection" if use_mass_collection else "test_drive"
    )

    # Initialize TuRBO if selected
    turbo_agent = None
    if use_turbo:
        print(f" {Style.CHECK} Initializing TuRBO-M Agent (Dim: {opt_dim})")
        turbo_agent = ASPECTOR_TurboM(
            dim=opt_dim,
            specs_weights=selected_weights,
            max_evals=n_max_evals,
            batch_size=turbo_batch_size,
            verbose=True
        )

    print()
    print(Style.DOUBLE_LINE)
    print(f"      STARTING PIPELINE | {n_workers} WORKERS       ".center(70))
    print(Style.DOUBLE_LINE)
    print()

    # Shared Execution Loop Variables
    total_completed = 0
    total_evals_target = n_max_evals
    
    try:
        if not use_turbo:
            # --- SOBOL MODE (One-shot batch) ---
            print(f" {Style.INFO} Mode: Sobol Exploration (One-shot)")
            samples = generator.generate(n_max_evals)
            
            # Inject Flags
            for s in samples:
                s['run_gatekeeper'] = 1
                s['run_full_char'] = 1

            for (completed, total, elapsed, data) in run_parallel_simulations(samples, extractor, n_workers):
                 rate = completed / elapsed if elapsed > 0 else 0
                 percent = (completed / total) * 100
                 bar = '#' * int(30 * completed // total) + '-' * (30 - int(30 * completed // total))
                 print(f" [{bar}] {percent:5.1f}% | {completed}/{total} | Rate: {rate:4.1f} sim/s", end='\r')
                 
                 if data and use_mass_collection and collector:
                     idx, result_val = data
                     # result_val is (reward, specs, full_res)
                     if len(result_val) == 3:
                         full_res = result_val[2]
                         # Log to collector
                         collector.log(full_res['parameters'], full_res['specs'], meta={'sim_id': full_res['id'], 'env': full_res['env']})

        else:
            # --- TURBO MODE (Iterative Loop) ---
            print(f" {Style.INFO} Mode: TuRBO-M Optimization Loop")
            
            while total_completed < n_max_evals:
                # 1. Ask
                curr_batch_size = min(turbo_batch_size, n_max_evals - total_completed)
                print(f"\n {Style.ARROW} TuRBO Asking for {curr_batch_size} candidates...")
                
                start_time_ask = time.time()
                X_next = turbo_agent.ask(curr_batch_size) # Tensor [batch, dim]
                # print(f" [Debug] Ask took {time.time()-start_time_ask:.2f}s")
                
                # 2. Map to Physical & Robust Expansion
                # "NeurIPS Quality": Adversarial Robustness Check.
                # Every candidate is simulated twice: 
                #   (A) Nominal Environment.
                #   (B) Randomized "Stress" Environment.
                # TuRBO receives the WORST performance of the two.
                
                X_list = X_next.tolist()
                
                # A. Nominal Samples
                samples_nom = generator.generate(curr_batch_size, u_samples=X_list, robust_env=False)
                # B. Robust Samples (Random Env)
                samples_rob = generator.generate(curr_batch_size, u_samples=X_list, robust_env=True)
                
                # Combine for execution (Batch Size * 2 sims)
                # Structure: [Nom_1, Nom_2, ... Nom_N, Rob_1, Rob_2, ... Rob_N]
                # We map indices: 
                #   Sim 1..N correspond to Candidate 1..N (Nominal)
                #   Sim N+1..2N correspond to Candidate 1..N (Robust)
                
                full_batch_samples = samples_nom + samples_rob
                total_sims = len(full_batch_samples)
                
                for s in full_batch_samples:
                    s['run_gatekeeper'] = 1
                    s['run_full_char'] = 1
                
                # 3. Evaluate (Parallel)
                print(f" {Style.ARROW} Simulating Robust Batch ({total_sims} sims for {curr_batch_size} candidates)...")
                
                sim_results = []
                for (b_completed, b_total, b_elapsed, data) in run_parallel_simulations(full_batch_samples, extractor, n_workers):
                    rate = b_completed / b_elapsed if b_elapsed > 0 else 0
                    print(f"   Batch Progress: {b_completed}/{total_sims} | Rate: {rate:.1f} sim/s", end='\r')
                    if data:
                        idx, result_val = data
                        sim_results.append(data) # Store for later map creation
                        
                        # Handle Mass Collection
                        if use_mass_collection and len(result_val) == 3:
                             full_res = result_val[2]
                             collector.log(full_res['parameters'], full_res['specs'], meta={'sim_id': full_res['id'], 'env': full_res['env']})
                
                # 4. Tell - Adversarial Aggregation
                # Maps: [0..N-1] -> Nominal, [N..2N-1] -> Robust
                # We need to construct a robust reward for each candidate i in [0..N-1]
                
                # Helper dictionary for O(1) lookup
                # sim_results holds (sim_idx, val_tuple)
                # val_tuple can be (rew, specs) or (rew, specs, full_res)
                # We normalize to (rew, specs) using slice [:2]
                results_map = {idx: val_tuple[:2] for idx, val_tuple in sim_results}
                
                ordered_specs = []
                valid_indices = []
                
                n_candidates = curr_batch_size
                
                for i in range(n_candidates):
                    # Indices for this candidate
                    idx_nom = i
                    idx_rob = i + n_candidates
                    
                    # Retrieve
                    res_nom = results_map.get(idx_nom)
                    res_rob = results_map.get(idx_rob)
                    
                    if res_nom and res_rob:
                         specs_nom = res_nom[1]
                         specs_rob = res_rob[1]
                         
                         # Check validity inside specs or assume valid if returned
                         # We use Worst-Case Reward implicitly by picking the specs set that corresponds to...
                         # Actually TuRBO computes reward inside `tell` using `scalarize`.
                         # We can't pass "Worst Reward" easily without pre-calculating it.
                         # Instead, we will pass the specs that are "Worse".
                         # But "Worse" depends on weights which are inside TuRBO.
                         # Simpler "NeurIPS" approach: We Average the specs? Or Worst-Case each spec?
                         # Worst-Case each Spec! 
                         # e.g. Gain = min(Gain_nom, Gain_rob), Power = max(Power_nom, Power_rob)
                         
                         worst_case_specs = {}
                         # Combine keys
                         all_keys = set(specs_nom.keys()) | set(specs_rob.keys())
                         
                         worst_case_specs['valid'] = specs_nom.get('valid', True) and specs_rob.get('valid', True)
                         
                         for key in all_keys:
                             if key == 'valid': continue
                             
                             val_n = specs_nom.get(key)
                             val_r = specs_rob.get(key)
                             
                             if val_n is None or val_r is None: continue # Skip if missing in one
                             
                             # Helper to compare
                             def is_better(v1, v2, k):
                                 # Returns True if v1 is "better" than v2
                                 # We want the WORSE one.
                                 if k in ["power", "integrated_noise", "settling_time", "vos", "thd", "ibias"]:
                                     return abs(v1) < abs(v2)
                                 else: # Gain, PM, UGBW etc -> Higher is better
                                     return v1 > v2

                             if is_better(val_n, val_r, key):
                                 worst_case_specs[key] = val_r # Pick Robust (Worse)
                             else:
                                 worst_case_specs[key] = val_n # Pick Nominal (Worse)
                        
                         ordered_specs.append(worst_case_specs)
                         valid_indices.append(i)
                    
                    elif res_nom:
                         # Fallback if Robust crashed but Nominal survived?
                         # Penalize? Or just take Nominal?
                         # NeurIPS strictness: Fail.
                         pass
                
                if len(valid_indices) > 0:
                     X_valid = X_next[valid_indices]
                     turbo_agent.tell(X_valid, ordered_specs)
                     
                     total_completed += len(valid_indices)
                     best_v = turbo_agent.state.best_value if turbo_agent.state.best_value is not None else 0.0
                     print(f"\n {Style.CHECK} Robust Batch (Worst-Case) processed. Total: {total_completed}/{n_max_evals} Best Cost: {best_v:.4f}")
                else:
                     print(f"\n {Style.X} All simulations in batch failed check.")

    except KeyboardInterrupt:
        print(f"\n\n {Style.X} Simulation interrupted for {netlist_name_base}.")
        
    finally:
        config_env.del_yaml()
        if collector:
            print(f" {Style.INFO} Flushing Data Collector...")
            collector.flush()
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
    
    selected_weights = {} 
    
    # Defaults
    run_config = {
        "n_workers": n_workers,
        "use_turbo": use_turbo,
        "n_max_evals": 1000,
        "turbo_batch_size": 64,
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
        
        persona = get_valid_int("Enter persona", 1, 5, 5)
        
        # ... Persona Dictionary Mapping ...
        if persona == 1: # Speed
            print(f" {Style.INFO} Persona: SPEED")
            selected_weights = {
                'ugbw': 10.0, 'slew_rate': 10.0, 'settling_time': 10.0, 
                'pm': 100.0, 'gm': 50.0, 'power': 5.0, 
                'gain_ol': 1.0 
            }
        elif persona == 2: # Precision
            print(f" {Style.INFO} Persona: PRECISION")
            selected_weights = {
                'gain_ol': 10.0, 'cmrr': 10.0, 'psrr': 10.0, 'vos': 10.0, 'thd': 10.0,
                'pm': 100.0, 'gm': 50.0, 'integrated_noise': 5.0, 
                'ugbw': 1.0 
            }
        elif persona == 3: # Efficiency
            print(f" {Style.INFO} Persona: EFFICIENCY")
            selected_weights = {
                'power': 20.0, 'integrated_noise': 10.0, 
                'pm': 100.0, 'gm': 50.0, 'ugbw': 5.0, 
                'gain_ol': 1.0 
            }
        elif persona == 4: # Compactness
            print(f" {Style.INFO} Persona: COMPACTNESS")
            selected_weights = {
                'area': 20.0, 
                'pm': 100.0, 'gm': 50.0, 
                'output_voltage_swing': 5.0, 'slew_rate': 5.0, 
                'ugbw': 1.0, 'gain_ol': 1.0 
            }
        else: # Balanced
            print(f" {Style.INFO} Persona: BALANCED (Default)")
            selected_weights = {
                 'gain_ol': 1.0, 'ugbw': 1.0, 'pm': 100.0, 'gm': 50.0,
                 'power': 2.0, 'vos': 5.0
            }

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
        run_config['selected_weights'] = selected_weights
        
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
    print("   2. Mass Collection (Parquet + .pt Graphs)")
    
    data_mode_sel = get_valid_int("Enter selection", 1, 2, 1)
    use_mass_collection = (data_mode_sel == 2)
    run_config['use_mass_collection'] = use_mass_collection
    
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
