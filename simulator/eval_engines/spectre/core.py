"""
core.py

Author: natelgrw
Last Edited: 01/15/2026

Core Spectre evaluation engine for circuit simulation and design optimization.
Handles netlist generation, simulation execution, result parsing, and 
cost function evaluation for iterative circuit optimization.
"""

import os
from jinja2 import Environment, FileSystemLoader
import os
import shutil
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
import yaml
import importlib
import random
import numpy as np
import uuid
import time
import gc
from simulator.eval_engines.spectre.parser import SpectreParser

debug = False 


# ===== Configuration Utilities ===== #


def get_config_info():
    """
    Retrieves configuration information from environment variables.

    Reads the BASE_TMP_DIR environment variable which specifies the base
    temporary directory for design generation and simulation.

    Returns:
    --------
    config_info : dict
        Dictionary containing configuration parameters from environment.
    """
    config_info = dict()
    base_tmp_dir = os.environ.get('BASE_TMP_DIR', None)
    if not base_tmp_dir:
        raise EnvironmentError('BASE_TMP_DIR is not set in environment variables')
    else:
        config_info['BASE_TMP_DIR'] = base_tmp_dir

    return config_info


# ===== Spectre Simulation Wrapper ===== #


class SpectreWrapper(object):
    """
    Wrapper for managing Spectre circuit simulations.

    Handles netlist generation from templates, simulation execution via Spectre,
    result parsing, and post-processing. Each instance manages one netlist and
    its associated simulation testbench.
    
    Initialization Parameters:
    --------------------------
    tb_dict : dict
        Testbench configuration dictionary.
    """

    def __init__(self, tb_dict):

        netlist_loc = tb_dict['netlist_template']
        if not os.path.isabs(netlist_loc):
            netlist_loc = os.path.abspath(netlist_loc)
        
        # load post-processing module and class
        pp_module = importlib.import_module(tb_dict['tb_module'])
        pp_class = getattr(pp_module, tb_dict['tb_class'])
        self.post_process = getattr(pp_class, tb_dict['post_process_function'])
        self.tb_params = tb_dict['tb_params']

        # get configuration information from environment
        self.config_info = get_config_info()

        self.root_dir = self.config_info['BASE_TMP_DIR']
        self.num_process = self.config_info.get('NUM_PROCESS', 1)

        # Calculate project root and lstp path relative to this file
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        self.lstp_path = os.path.join(self.project_root, "lstp")

        # create unique design directory name
        _, dsn_netlist_fname = os.path.split(netlist_loc)
        # Use UUID to ensure unique design directory for every instance/process
        self.base_design_name = os.path.splitext(dsn_netlist_fname)[0] + "_" + uuid.uuid4().hex
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.gen_dir, exist_ok=True)

        # setup jinja2 template environment
        file_loader = FileSystemLoader(os.path.dirname(netlist_loc))
        self.jinja_env = Environment(loader=file_loader)
        self.template = self.jinja_env.get_template(dsn_netlist_fname)

    def _get_design_name(self, state):
        """
        Creates a unique identifier filename based on design state parameters.

        Parameters:
        -----------
        state : dict
            Dictionary of parameter names and values for the design.
        
        Returns:
        --------
        fname : str
            Unique filename identifier for the design.
        """
        fname = self.base_design_name

        # append scaled values for each parameter
        for value in state.values():
            if value<=2E-13:
                x = value*1E14
                fname += "_" + str(round(x,2))
            elif value<=1E-6:
                x = value*1E7
                fname += "_" + str(round(x,2))
            else:
                fname += "_" + str(round(value,2))

        return fname

    def _create_design(self, state, new_fname):
        """
        Creates sized netlist file from template and design parameters.

        Parameters:
        -----------
        state : dict
            Dictionary of parameter values for rendering the template.
        new_fname : str
            Filename for the generated netlist (without extension).
        
        Returns:
        --------
        design_folder : str
            Path to the folder containing the generated netlist.
        fpath : str
            Full path to the generated netlist file.
        """
        # render template with design parameters
        render_context = state.copy()
        
        # Ensure Critical Environment Variables are set!
        # If they came in via state, they are already there.
        # If not, we should probably set defaults, but ideally 
        # they must be in 'state' from the generator.
        
        # NOTE: Jinja templates fail silently sometimes or create invalid netlists
        # if variables are missing.
        
        render_context['lstp_path'] = self.lstp_path
        
        output = self.template.render(**render_context)
        design_folder = os.path.join(self.gen_dir, new_fname)
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.scs')

        with open(fpath, 'w') as f:
            f.write(output)
            f.close()

        return design_folder, fpath

    def _simulate(self, fpath):
        """
        Executes Spectre simulation on generated netlist.

        Parameters:
        -----------
        fpath : str
            Full path to the netlist file to simulate.
        
        Returns:
        --------
        info : int
            Error code. 0 indicates successful simulation, 1 indicates error.
        """
        # construct Spectre command
        command = ['spectre', '%s'%fpath, '-format', 'psfbin' ,'> /dev/null 2>&1']
        log_file = os.path.join(os.path.dirname(fpath), 'log.txt')
        err_file = os.path.join(os.path.dirname(fpath), 'err_log.txt')

        # execute simulation and capture output
        with open(log_file, 'w') as file1, open(err_file,'w') as file2:
          exit_code = subprocess.call(command, cwd=os.path.dirname(fpath), stdout=file1, stderr=file2)
        file1.close()
        file2.close()

        # determine success based on exit code
        info = 0
        if (exit_code % 256):
            info = 1

        return info

    def _create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        """
        Creates a design netlist and runs simulation.

        Parameters:
        -----------
        state : dict
            Dictionary of design parameter values.
        dsn_name : str, optional
            Custom design name. If None, auto-generated from state.
        verbose : bool
            If True, prints design name information.
        
        Returns:
        --------
        state : dict
            The input design state.
        specs : dict
            Dictionary of post-processed simulation results.
        info : int
            Error code from simulation (0 = success).
        """
        # generate design name if not provided
        if dsn_name == None:
            dsn_name = self._get_design_name(state)
        else:
            dsn_name = str(dsn_name)
    
        if verbose:
            print('dsn_name', dsn_name)

        # create netlist from template and run simulation
        design_folder, fpath = self._create_design(state, dsn_name)
        
        try:
            info = self._simulate(fpath)
            results = self._parse_result(design_folder)
            
            # post process results
            if self.post_process:
                specs = self.post_process(results, state)
                return state, specs, info
            specs = results
            
            return state, specs, info
        finally:
            self._cleanup(design_folder)

    def _cleanup(self, design_folder):
        """
        Removes the generated design folder and all its contents to save space.
        """
        gc.collect() # Force close of any open file handles from libpsf
        time.sleep(1.0) 

        if os.path.exists(design_folder):
            try:
                shutil.rmtree(design_folder, ignore_errors=False)
            except Exception as e:
                # Fallback to stronger delete
                subprocess.call(['rm', '-rf', design_folder])

    def _parse_result(self, design_folder):
        """
        Parses simulation results from Spectre output files.

        Parameters:
        -----------
        design_folder : str
            Path to the design folder containing simulation results.
        
        Returns:
        --------
        res : dict
            Dictionary of parsed simulation results.
        """
        # extract folder name and locate raw simulation output
        _, folder_name = os.path.split(design_folder)
        raw_folder = os.path.join(design_folder, '{}.raw'.format(folder_name))

        # parse results
        res = SpectreParser.parse(raw_folder)       

        return res

    def run(self, states, design_names=None, verbose=False):
        """
        Executes simulations for multiple design states in parallel.

        Uses thread pool to run multiple simulations concurrently based on
        configured number of processes.

        Parameters:
        -----------
        states : list
            List of design state dictionaries to simulate.
        design_names : list, optional
            Custom design names for each state. If None, auto-generated names used.
        verbose : bool
            If True, prints design name information during execution.
        
        Returns:
        --------
        specs : list
            List of (state, specs, info) tuples for each design simulated.
        """
        # execute simulations in parallel using thread pool
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self._create_design_and_simulate, arg_list)
        pool.close()

        return specs

    def return_path(self):
        """
        Returns the design generation directory path.

        Returns:
        --------
        str
            Path to the directory containing generated designs.
        """
        return self.gen_dir


# ===== Circuit Evaluation Engine ===== #


class EvaluationEngine(object):
    """
    Main evaluation engine for circuit optimization.
    Stripped down for simple data generation.

    Initialization Parameters:
    --------------------------
    yaml_fname : str
        Path to YAML configuration file specifying parameters,
        specifications, and testbench setup.
    """

    def __init__(self, yaml_fname):

        self.design_specs_fname = yaml_fname
        
        # load configuration from YAML file
        with open(yaml_fname, 'r') as f:
            self.ver_specs = yaml.load(f, Loader=yaml.Loader)
        f.close()

        # setup testbench modules
        self.measurement_specs = self.ver_specs['measurement']
        tbs = self.measurement_specs['testbenches']
        self.netlist_module_dict = {}
        for tb_kw, tb_val in tbs.items():
            self.netlist_module_dict[tb_kw] = SpectreWrapper(tb_val)

    def evaluate(self, design_list, debug=True, parallel_config=None):
        """
        Evaluates designs and returns processed results.

        Parameters:
        -----------
        design_list : list of dicts
            List of design parameter dictionaries.
        debug : bool
        
        Returns:
        --------
        results : list
            List of (state, specs, info) tuples.
        """
        results = []
        
        for state in design_list:
            # run simulations for all testbenches
            sim_results = {}
            for netlist_name, netlist_module in self.netlist_module_dict.items():
                sim_results[netlist_name] = netlist_module._create_design_and_simulate(state, verbose=debug)

            # get specifications from results (using subclass logic)
            # subclass (e.g. OpampMeasMan) must implement get_specs
            if hasattr(self, 'get_specs'):
                specs_dict = self.get_specs(sim_results, state)
            else:
                # If no post-processing mapping, just return the raw specs from wrapper
                # Assuming single testbench for simplicity if no get_specs
                first_res = list(sim_results.values())[0]
                specs_dict = first_res[1]

            # Return format matching legacy expectation: (state, specs, info)
            # We construct a dummy info here
            info = 0
            results.append((state, specs_dict, info))

        return results

