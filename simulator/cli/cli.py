"""
cli.py

Author: natelgrw
Last Edited: 01/24/2026

Command Line Interface for ASPECTOR Core.
Provides interactive setup for running circuit optimization pipelines.
"""

import os
import sys
import argparse
import numpy as np
import time

# Add workspace to path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from simulator import globalsy
from optimization.turbo_1 import Turbo1
from optimization.turbo_m import TurboM
from simulator.eval_engines.utils.netlist_to_graph import parse_netlist_to_graph, extract_sizing_map
from simulator.eval_engines.spectre.configs.config_env import EnvironmentConfig
from simulator.eval_engines.extractor.extractor import Extractor, extract_parameter_names, build_bounds, classify_opamp_type

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

def get_valid_int(prompt_text, min_val=None, max_val=None):
    print(f" {prompt_text}")
    if min_val is not None and max_val is not None:
        print(f"    Bounds:  {min_val} to {max_val}")
    print()

    while True:
        prompt = f" {Style.ARROW} Enter value: "
        user_input = input(prompt).strip()
        
        if not user_input:
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

def get_valid_float(prompt_text, min_val=None, max_val=None):
    print(f" {prompt_text}")
    if min_val is not None and max_val is not None:
        print(f"    Bounds:  {min_val} to {max_val}")
    print()

    while True:
        prompt = f" {Style.ARROW} Enter value: "
        user_input = input(prompt).strip()
        
        if not user_input:
            continue

        try:
            val = float(user_input)
        except ValueError:
            print_error("Please enter a valid number")
            continue

        if min_val is not None and val < min_val:
            print_error(f"Value must be >= {min_val}")
            continue
        if max_val is not None and val > max_val:
            print_error(f"Value must be <= {max_val}")
            continue
        
        print() 
        return val

def get_valid_choice(prompt_text, choices):
    valid_str = ", ".join(str(c) for c in choices)
    print(f" {prompt_text}")
    print(f"    Options: {valid_str}")
    print()

    while True:
        prompt = f" {Style.ARROW} Enter value: "
        user_input = input(prompt).strip()
        
        if not user_input:
            continue
            
        try:
            # Try converting to int if choices are ints
            if all(isinstance(c, int) for c in choices):
                val = int(user_input)
            else:
                val = user_input
                
            if val in choices:
                print()
                return val
            else:
                print_error(f"Invalid choice. Must be one of: {valid_str}")
        except ValueError:
            print_error(f"Invalid format")

def main():
    clear_screen()
    print_header()

    # 1. Netlist Selection
    netlist_name_base, scs_file_path = get_netlist_input()

    # 2. Technology Setup
    print_section("Technology Setup")
    
    fet_num = get_valid_choice(
        "Transistor Card Size (nm)", 
        choices=[7, 10, 14, 16, 20]
    )

    # 3. Operating Conditions
    print_section("Operating Conditions")

    # Set appropriate VDD range for FinFET / advanced nodes
    # These technologies typically operate between 0.6V and 1.0V nominal
    vdd = get_valid_float(
        "Supply Voltage (VDD)", 
        min_val=0.5, 
        max_val=1.2
    )
    
    vcm = get_valid_float(
        "Common Mode Voltage (VCM)", 
        min_val=0.0, 
        max_val=vdd
    )
    
    tempc = get_valid_float(
        "Temperature (C)", 
        min_val=-55.0, 
        max_val=125.0
    )

    # 4. Algorithm Selection
    print_section("Optimization Algorithm")
    print("   1. TuRBO (Trust Region Bayesian Optimization)")
    print()
    
    while True:
        algo_choice = input(f" {Style.ARROW} Enter selection: ").strip()
        if not algo_choice:
            continue
        elif algo_choice == '1':
            algo = "turbo"
            break
        else:
             print_error("Invalid selection.")

    print_success(f"Algorithm selected: {algo.upper()}")
    
    max_evals = get_valid_int(
        "Optimization Iterations",
        min_val=21,
        max_val=10000
    )

    # 4b. Specification Selection
    print_section("Specification Selection")
    
    # Defaults from globalsy (metadata: value, weight, sim_type)
    defaults = globalsy.spec_metadata
    
    # 1. Select Measurements
    print(" Select specifications to MEASURE (available for viewing):")
    measured_specs = []
    
    # Group by simulation type for clarity
    sim_types = {0: "AC", 1: "DC", 2: "NOISE", 3: "TRANSIENT"}
    grouped_specs = {}
    for k, v in defaults.items():
        st = v[2]
        if st not in grouped_specs: grouped_specs[st] = []
        grouped_specs[st].append(k)
        
    for st_code in sorted(grouped_specs.keys()):
        print(f" -- {sim_types[st_code]} Analysis --")
        for spec in grouped_specs[st_code]:
            while True:
                ans = input(f"    Measure '{spec}'? [y/N]: ").strip().lower()
                if ans == 'y':
                    measured_specs.append(spec)
                    break
                elif ans == 'n' or ans == '':
                    break
    
    if not measured_specs:
        print_error("No specs selected. Selecting 'gain' and 'power' by default.")
        measured_specs = ['gain', 'power']

    # 2. Select Optimization Targets
    print("\n Select specifications to OPTIMIZE (from measured):")
    optimized_specs = []
    specs_ideal = {}
    specs_weights = {}
    
    for spec in measured_specs:
        while True:
            ans = input(f"    Optimize '{spec}'? [y/N]: ").strip().lower()
            if ans == 'y':
                optimized_specs.append(spec)
                
                # Get Target
                def_val = defaults[spec][0]
                val_in = input(f"       Target Value [{def_val}]: ").strip()
                specs_ideal[spec] = float(val_in) if val_in else def_val
                
                # Get Weight
                def_w = defaults[spec][1]
                w_in = input(f"       Weight [{def_w}]: ").strip()
                specs_weights[spec] = float(w_in) if w_in else def_w
                break
            elif ans == 'n' or ans == '':
                break

    if not optimized_specs:
        print_error("No optimization specs selected. Defaulting to first measured spec.")
        s = measured_specs[0]
        optimized_specs.append(s)
        specs_ideal[s] = defaults[s][0]
        specs_weights[s] = defaults[s][1]

    # Determine required simulations
    sim_flags = {'ac': False, 'dc': False, 'noise': False, 'tran': False}
    for spec in measured_specs:
        st = defaults[spec][2]
        if st == 0: sim_flags['ac'] = True
        elif st == 1: sim_flags['dc'] = True
        elif st == 2: sim_flags['noise'] = True
        elif st == 3: sim_flags['tran'] = True

    print(f"\n {Style.CHECK} Configured Simulations: {[k.upper() for k,v in sim_flags.items() if v]}")

    # 5. Pipeline Execution Setup
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    results_dir = os.path.join(project_root, "results", netlist_name_base)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        os.makedirs(os.path.join(results_dir, "graph"))
        os.makedirs(os.path.join(results_dir, "sizings"))
        os.makedirs(os.path.join(results_dir, "specs"))
        os.makedirs(os.path.join(results_dir, "op_points"))

    print_section("Initialization")
    # print_info(f"Output Directory: {results_dir}")
    
    print(f" {Style.CHECK} Generating Graph Representation")
    parse_netlist_to_graph(scs_file_path, os.path.join(results_dir, "graph"), topology_id=1)
    size_map = extract_sizing_map(scs_file_path)

    np.random.seed(2000)

    print(f" {Style.CHECK} Configuring Environment")
    
    # Load specs from global config (could be made dynamic later)
    # Ensure these exist in your globalsy.py
    # specs_dict = globalsy.specs_dict 
    shared_ranges = globalsy.shared_ranges

    params_id = extract_parameter_names(scs_file_path)
    # The build_bounds call is dynamically determined in extractor, 
    # ensuring that logic resides there.
    full_lb, full_ub, opt_lb, opt_ub, opt_params = build_bounds(params_id, shared_ranges, vdd, vcm, tempc, fet_num)

    opamp_type = classify_opamp_type(scs_file_path)
    
    # reformatted specs for extractor input (using OPTIMIZED specs)
    specs_id = optimized_specs
    specs_ideal_list = [specs_ideal[s] for s in specs_id]
    specs_weights_list = [specs_weights[s] for s in specs_id]

    # For EnvironmentConfig, we pass measured specs so the meas_man knows what to look for
    # (Though currently meas_man might look for everything hardcoded, passing empty dict might break it 
    # if it relies on keys, so let's pass dummy targets for measured specs not in optimizer)
    env_specs = {}
    for s in measured_specs:
        env_specs[s] = specs_ideal.get(s, 0.0) # 0.0 or default if not optimized

    config_env = EnvironmentConfig(scs_file_path, opamp_type, env_specs, params_id, full_lb, full_ub)
    yaml_path = config_env.write_yaml_configs()

    extractor = Extractor(
        dim=len(opt_lb),
        opt_params=opt_params,
        params_id=params_id,
        specs_id=specs_id,
        specs_ideal=specs_ideal_list,
        specs_weights=specs_weights_list, # New Arg
        sim_flags=sim_flags,              # New Arg
        vcm=vcm,
        vdd=vdd,
        tempc=tempc,
        ub=opt_ub,
        lb=opt_lb,
        yaml_path=yaml_path,
        fet_num=fet_num,
        results_dir=results_dir,
        netlist_name=netlist_name_base,
        size_map=size_map
    )

    print(f" {Style.CHECK} Initializing TuRBO Optimizer")
    optimizer = Turbo1(
        f=extractor,                   # Objective function handle
        lb=opt_lb,                     # Lower bounds array (optimized params only)
        ub=opt_ub,                     # Upper bounds array (optimized params only)
        n_init=20,                     # Initial design points from Latin hypercube
        max_evals=max_evals,            # Maximum total evaluations allowed
        batch_size=5,                  # Points evaluated per batch
        verbose=True,                  # Print batch-level progress
        use_ard=True,                  # Automatic Relevance Determination for GP
        max_cholesky_size=2000,        # Cholesky vs Lanczos threshold
        n_training_steps=30,           # ADAM optimization steps per batch
        min_cuda=10.40,                # CUDA memory threshold
        device="cpu",                  # Compute device
        dtype="float32",               # Floating point precision
    )

    print()
    print(Style.DOUBLE_LINE)
    print("               STARTING OPTIMIZATION LOOP               ".center(70))
    print(Style.DOUBLE_LINE)
    print()

    try:
        optimizer.optimize()

    except KeyboardInterrupt:
        print(f"\n\n {Style.X} Optimization interrupted by user.")
    finally:
        config_env.del_yaml()
        print(f"\n {Style.CHECK} Pipeline finished.")

if __name__ == "__main__":
    main()