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
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
import yaml
import importlib
import random
import numpy as np
from simulator.eval_engines.utils.design_reps import IDEncoder, Design
from simulator.eval_engines.spectre.parser import SpectreParser
import shutil

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
        self.base_design_name = os.path.splitext(dsn_netlist_fname)[0] + str(random.randint(0,10000))
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
        info = self._simulate(fpath)
        results = self._parse_result(design_folder)
        
        # post process results
        if self.post_process:
            specs = self.post_process(results, self.tb_params)
            self._cleanup(design_folder)
            return state, specs, info
        specs = results
        self._cleanup(design_folder)
        
        return state, specs, info

    def _cleanup(self, design_folder):
        """
        Removes the generated design folder and all its contents to save space.
        Uses ignore_errors=True to avoid crashing on NFS holdover files (.nfs*).
        """
        if os.path.exists(design_folder):
            try:
                shutil.rmtree(design_folder, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to cleanup {design_folder}: {e}")

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

    Handles design space exploration, cost function computation, and
    evaluation of circuit designs against performance specifications.
    Manages parameter space discretization and design ID encoding.

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
            # print("DEBUG ver_specs =", type(self.ver_specs), self.ver_specs)
        f.close()

        # extract specification ranges
        self.spec_range = self.ver_specs['spec_range']
        params = self.ver_specs['params']

        # discretize parameter space into vectors
        self.params_vec = {}
        self.search_space_size = 1
        for key, value in params.items():
            if value[0] == value[1]:
                self.params_vec[key] = [value[0]]
            else:
                self.params_vec[key] = np.arange(value[0], value[1], value[2]).tolist()
            self.search_space_size = self.search_space_size * len(self.params_vec[key])

        # compute parameter bounds
        self.params_min = [0]*len(self.params_vec)
        self.params_max = []
        for val in self.params_vec.values():
            self.params_max.append(len(val)-1)

        # setup ID encoder and testbench modules
        self.id_encoder = IDEncoder(self.params_vec)
        self.measurement_specs = self.ver_specs['measurement']
        tbs = self.measurement_specs['testbenches']
        self.netlist_module_dict = {}
        for tb_kw, tb_val in tbs.items():
            self.netlist_module_dict[tb_kw] = SpectreWrapper(tb_val)

    @property
    def num_params(self):
        """
        Gets the number of parameters in the search space.

        Returns:
        --------
        int
            Number of design parameters.
        """
        return len(self.params_vec)

    def generate_data_set(self, n=1, debug=False):
        """
        Generates n valid design samples from the search space.

        Randomly samples designs and evaluates them until n valid designs
        are found. A valid design meets all specification constraints.

        Parameters:
        -----------
        n : int
            Number of valid designs to generate.
        debug : bool
            If True, raises exceptions instead of suppressing errors.
        parallel_config : dict, optional
            Configuration for parallel evaluation.
        
        Returns:
        --------
        valid_designs : list
            List of n valid Design objects with evaluated specs and costs.
        
        Raises:
        -------
        ValueError
            If unable to find n valid designs after extensive random sampling.
        """
        valid_designs, tried_designs = [], []
        nvalid_designs = 0 
        useless_iter_count = 0

        # randomly sample design from parameter space
        while len(valid_designs) < n:
            design = {}
            for key, vec in self.params_vec.items():
                rand_idx = random.randrange(len(vec))
                design[key] = rand_idx
            design = Design(self.spec_range, self.id_encoder, list(design.values()))
            
            # skip if design already tried
            if design in tried_designs:
                if (useless_iter_count > n * 5):
                    raise ValueError("Random selection of a fraction of search space did not "
                                     "result in {} number of valid designs".format(n))
                useless_iter_count += 1
                continue
            
            # evaluate design
            design_result = self.evaluate([design], debug=debug)[0]
            if design_result['valid']:
                design.cost = design_result['cost']
                for key in design.specs.keys():
                    design.specs[key] = design_result[key]
                valid_designs.append(design)
            else:
                nvalid_designs += 1
            tried_designs.append(design)

        return valid_designs[:n]

    def evaluate(self, design_list, debug=True, parallel_config=None):
        """
        Evaluates designs and returns processed results.

        Executes simulations for each design and computes cost function.
        Errors are suppressed unless debug mode is enabled.

        Parameters:
        -----------
        design_list : list
            List of Design objects to evaluate.
        debug : bool
            If True, raises exceptions. If False, returns {'valid': False}.
        parallel_config : dict, optional
            Configuration for parallel evaluation.
        
        Returns:
        --------
        results : list
            List of result dictionaries with cost and spec values.
        """
        results = []
        
        if len(design_list) > 1:
            for design in design_list:
                try:
                    result = self._evaluate(design)
                except Exception as e:
                    if debug:
                        raise e
                    result = {'valid': False}
                    print(getattr(e, 'message', str(e)))
                results.append(result)
        else:
            try:
                netlist_name, netlist_module = list(self.netlist_module_dict.items())[0]
                result = netlist_module._create_design_and_simulate(design_list[0])
            except Exception as e:
                if debug:
                    raise e
                result = {'valid': False}
                print(getattr(e, 'message', str(e)))
            results.append(result)

        return results

    def _evaluate(self, design):
        """
        Internal evaluation of a single design.

        Maps design ID indices to actual parameter values, runs simulations,
        and computes cost function from results.

        Parameters:
        -----------
        design : Design
            Design object to evaluate.
        
        Returns:
        --------
        specs_dict : dict
            Dictionary containing all specs and computed cost.
        """
        # map design indices to actual parameter values
        state_dict = dict()
        design_i = 0
        for key, vec in self.params_vec.items():
            if len(vec) == 1:
                state_dict[key] = vec[0]
            else:
                state_dict[key] = vec[design.id[design_i]]
                design_i += 1

        dsn_names = [design.id]

        # run simulations for all testbenches
        results = {}
        for netlist_name, netlist_module in self.netlist_module_dict.items():
            results[netlist_name] = netlist_module.create_design_and_simulate(state_dict, dsn_names)

        # get specifications from results
        specs_dict = self.get_specs(results, self.measurement_specs['meas_params'])        
        specs_dict['cost'] = self.cost_fun(specs_dict)

        return specs_dict
