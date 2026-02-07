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
# from functools import partial # Removed as part of cleanup

# Add workspace to path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from simulator import globalsy
from simulator.compute.runner import run_parallel_simulations
from algorithms.sobol.generator import SobolSizingGenerator
from simulator.eval_engines.utils.netlist_to_graph import parse_netlist_to_graph, extract_sizing_map
from simulator.eval_engines.spectre.configs.config_env import EnvironmentConfig
from simulator.eval_engines.extractor.extractor import Extractor, extract_parameter_names, classify_opamp_type

# ASCII Art and Styles
class Style:
    # Use simple unicode/ascii characters instead of colors
    CHECK = "[+]"
    X = "[!]"
    INFO = "[i]"
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

def get_netlist_input():
    print_section("Netlist Selection")
    
    # List available netlists
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    base_path = os.path.join(project_root, "demo_netlists")
    
    if os.path.exists(base_path):
        print(" Available Netlists:")
        files = [f[:-4] for f in os.listdir(base_path) if f.endswith('.scs')]
        for i, f in enumerate(sorted(files)):
            print(f"   {i+1}. {f}")
        print()

    while True:
        prompt = f" {Style.ARROW} Enter netlist name: "
        netlist_name = input(prompt).strip()
        
        if not netlist_name:
            continue

        # Check if file exists in demo_netlists (adding extension if missing)
        if not netlist_name.endswith('.scs'):
            full_path = os.path.join(base_path, netlist_name + ".scs")
        else:
            full_path = os.path.join(base_path, netlist_name)
            netlist_name = netlist_name[:-4] 
            
        if os.path.exists(full_path):
            print_success(f"Netlist loaded: {netlist_name}")
            return netlist_name, full_path
        else:
            print_error(f"File not found: {netlist_name}")

def get_valid_int(prompt_text, min_val=None, max_val=None, default=None):
    print(f" {prompt_text}")
    if min_val is not None and max_val is not None:
        print(f"    Bounds:  {min_val} to {max_val}")
    if default is not None:
        print(f"    Default: {default}")
    print()

    while True:
        prompt = f" {Style.ARROW} Enter value: "
        user_input = input(prompt).strip()
        
        if not user_input:
            if default is not None:
                return default
            continue

        try:
            val = int(user_input)
        except ValueError:
            print_error("Please enter a valid integer")
            continue

        if min_val is not None and val < min_val:
            print_error(f"Value must be >= {min_val}")
            continue
        if max_val is not None and val > max_val:
            print_error(f"Value must be <= {max_val}")
            continue
        
        print() 
        return val

# worker_task moved to simulator/compute/runner.py

def main():
    clear_screen()
    print_header()

    # 1. Netlist Selection
    netlist_name_base, scs_file_path = get_netlist_input()

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

    # 3. Algorithm (Automatic)
    print_section("Algorithm Selection")
    print(f" {Style.INFO} Using SOBOL Sequence Generator (Low-Discrepancy Sampling)")
    print(f" {Style.INFO} Environment Parameters (VDD, Temp, etc.) will be sampled automatically.")

    # 4. Sample Count
    n_samples = get_valid_int(
        "Number of Samples to Generate",
        min_val=1,
        max_val=1000000
    )

    # 5. Pipeline Execution Setup
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    results_dir = os.path.join(project_root, "results", netlist_name_base)

    sims_dir = os.path.join(results_dir, "simulations")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        os.makedirs(os.path.join(results_dir, "graph"))
        os.makedirs(sims_dir) # Use NEW unified simulations directory
    else:
        # Ensure subdirs exist if re-running
        os.makedirs(os.path.join(results_dir, "graph"), exist_ok=True)
        os.makedirs(sims_dir, exist_ok=True)

    print_section("Initialization")
    
    print(f" {Style.CHECK} Generating Graph Representation")
    parse_netlist_to_graph(scs_file_path, os.path.join(results_dir, "graph"), sim_id=1, topology_id=netlist_name_base)
    size_map = extract_sizing_map(scs_file_path)

    # Extract parameters from netlist
    params_id = extract_parameter_names(scs_file_path)
    
    # Filter out environment params handled internally by generators
    ignored_params = ['fet_num', 'vdd', 'vcm', 'tempc', 'rfeedback_val', 'rsrc_val', 'cload_val']
    sizing_params_for_gen = [p for p in params_id if p not in ignored_params]
    
    print(f" {Style.CHECK} Initializing Sobol Generator")
    # Removed fixed seed=42 to allow unique samples on every run
    generator = SobolSizingGenerator(sizing_params_for_gen, seed=None)
    
    print(f" {Style.CHECK} Generating {n_samples} Samples...")
    samples = generator.generate(n_samples)

    # Inject Simulation Control Flags for Full Characterization
    for s in samples:
        s['run_gatekeeper'] = 1
        s['run_full_char'] = 1

    print(f" {Style.CHECK} Generation Complete.")

    # 6. Simulation Loop Prep
    print_section("Simulation Execution")
    
    # Configure Extractor
    sim_flags = {'ac': True, 'dc': True, 'noise': True, 'tran': True}
    
    opamp_type = classify_opamp_type(scs_file_path)
    
    # Config ENV
    full_lb = [-1e9] * len(params_id)
    full_ub = [ 1e9] * len(params_id)
    
    config_env = EnvironmentConfig(scs_file_path, opamp_type, {}, params_id, full_lb, full_ub)
    yaml_path = config_env.write_yaml_configs()
    print(f" {Style.INFO} Configuration Written to: {yaml_path}")
    
    # Initialize Extractor
    extractor = Extractor(
        dim=len(params_id),
        opt_params=params_id,
        params_id=params_id,
        specs_id=[],            
        specs_ideal=[],
        specs_weights=[],
        sim_flags=sim_flags,
        vcm=0,                 
        vdd=0,                 
        tempc=27,              
        ub=full_ub,            
        lb=full_lb,            
        yaml_path=yaml_path,
        fet_num=0,             
        results_dir=results_dir, # This now points to cleaner structure
        netlist_name=netlist_name_base,
        size_map=size_map
    )

    print()
    print(Style.DOUBLE_LINE)
    print(f"      STARTING BATCH: {n_samples} SAMPLES | {n_workers} WORKERS       ".center(70))
    print(Style.DOUBLE_LINE)
    print()

    try:
        # Run using the new compute runner
        # It yields (completed, total, elapsed) for progress updates
        
        for (completed, total, elapsed) in run_parallel_simulations(samples, extractor, n_workers):
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total - completed) / rate if rate > 0 else 0
            
            # Simple Progress Bar
            percent = (completed / total) * 100
            bar_len = 30
            filled_len = int(bar_len * completed // total)
            bar = '#' * filled_len + '-' * (bar_len - filled_len)
            
            print(f" [{bar}] {percent:5.1f}% | {completed}/{total} | Rate: {rate:4.1f} sim/s | ETA: {remaining/60:4.1f} min ", end='\r')
            
    except KeyboardInterrupt:
        print(f"\n\n {Style.X} Simulation interrupted by user.")
        
    finally:
        # Cleanup
        config_env.del_yaml() 
        
        end_time = time.time()
        # Ensure start_time is defined if we crashed before loop
        # But we can just use elapsed from the loop variables effectively or calc here
        # Note: 'elapsed' variable scope is inside loop, might be undefined if loop never ran.
        # But end_time - start_time? start_time was inside runner.
        # Let's just say finished.
        
        print(f"\n\n {Style.CHECK} Pipeline finished.")
        print(f" {Style.INFO} Results saved to: {sims_dir}")

if __name__ == "__main__":
    main()
