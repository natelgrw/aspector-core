"""
core.py

Author: natelgrw
Last Edited: 04/01/2026

Core Spectre evaluation engine for simulation orchestration.

Handles netlist generation, simulation execution, result parsing, and
measurement post-processing for iterative design evaluation.
"""

import os
import tempfile
from jinja2 import Environment, FileSystemLoader
import shutil
import subprocess
# Programmatic configuration dictionaries are used (YAML not required)
import importlib
import json
import hashlib
import uuid
import time
import gc
from shutil import which
from simulator.eval_engines.spectre.parser import SpectreParser
from simulator.eval_engines.utils.design_reps import extract_sizing_map


# ===== Spectre Simulation Wrapper ===== #


class SpectreWrapper:
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

        # build model-library root relative to project root
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        # Keep the historical variable name for template compatibility.
        # In this repo the actual PTM model tree lives under project_root/ptm.
        self.lstp_path = os.path.join(self.project_root, "ptm")
        self.ptm_path = self.lstp_path

        # create scratch directory in system temp
        _, dsn_netlist_fname = os.path.split(netlist_loc)
        self.base_design_name = os.path.splitext(dsn_netlist_fname)[0] + "_" + uuid.uuid4().hex
        self.gen_dir = tempfile.mkdtemp(prefix="aspector_", suffix="_" + self.base_design_name)

        # set up jinja2 template environment
        file_loader = FileSystemLoader(os.path.dirname(netlist_loc))
        self.jinja_env = Environment(loader=file_loader)
        self.template = self.jinja_env.get_template(dsn_netlist_fname)

        # Build sizing map once so post-process can initialize per-MM values
        # even when DCOP fails.
        try:
            self.size_map = extract_sizing_map(netlist_loc)
        except Exception:
            self.size_map = {}

    def _get_design_name(self, state):
        """
        Creates a unique identifier filename based on design state.

        Parameters:
        -----------
        state (dict): Dictionary of parameter names and values for the design.
        
        Returns:
        --------
        fname (str): Unique filename identifier for the design.
        """
        # use a short hash to avoid long filenames
        try:
            s = json.dumps(state, sort_keys=True, default=str)
        except Exception:
            s = str(state)
        short_id = hashlib.sha1(s.encode()).hexdigest()[:12]
        fname = f"{self.base_design_name}_{short_id}"
        return fname

    def _create_design(self, state, new_fname):
        """
        Create sized netlist file from template and design parameters.

        Parameters:
        -----------
        state (dict): Dictionary of parameter values for rendering the template.
        new_fname (str): Filename for the generated netlist (without extension).
        
        Returns:
        --------
        design_folder (str): Path to the folder containing the generated netlist.
        fpath (str): Full path to the generated netlist file.
        """
        # render template with design parameters
        render_context = state.copy()

        render_context['lstp_path'] = self.lstp_path
        render_context['ptm_path'] = self.ptm_path
        
        output = self.template.render(**render_context)

        # create a short, safe folder/file name derived from state
        try:
            s = json.dumps(state, sort_keys=True, default=str)
        except Exception:
            s = str(state)
        short_id = hashlib.sha1(s.encode()).hexdigest()[:12]

        safe_base = self.base_design_name
        folder_name = f"{safe_base}_{short_id}"
        design_folder = os.path.join(self.gen_dir, folder_name)
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, f"{safe_base}_{short_id}.scs")

        # write netlist
        with open(fpath, 'w') as f:
            f.write(output)

        return design_folder, fpath

    def _simulate(self, fpath):
        """
        Execute Spectre simulation on generated netlist.

        Parameters:
        -----------
        fpath (str): Full path to the netlist file to simulate.
        
        Returns:
        --------
        info (int): Error code. 0 indicates successful simulation, 1 indicates error.
        """
        # construct spectre command
        command = ['nice', '-n', '19', 'spectre', '%s'%fpath, '-format', 'psfbin']
        log_file = os.path.join(os.path.dirname(fpath), 'log.txt')
        err_file = os.path.join(os.path.dirname(fpath), 'err_log.txt')

        # fail fast with explicit diagnostics when Spectre is not on PATH
        if which('spectre') is None:
            try:
                with open(err_file, 'a') as f:
                    f.write('[ENV] spectre executable not found on PATH.\n')
            except Exception:
                pass
            print('   [!] Spectre executable not found on PATH.')
            return 1

        # get timeout from testbench params; default 5 minutes (300s)
        sim_timeout = self.tb_params.get('timeout', 300)

        # execute simulation with timeout; catch hung processes
        info = 0
        try:
            with open(log_file, 'w') as file1, open(err_file, 'w') as file2:
                exit_code = subprocess.run(command, cwd=os.path.dirname(fpath), stdout=file1, stderr=file2, timeout=sim_timeout).returncode
            if (exit_code % 256):
                info = 1
                try:
                    with open(err_file, 'a') as f:
                        f.write(f'\n[EXIT] spectre exited with code {exit_code}.\n')
                except Exception:
                    pass
        except subprocess.TimeoutExpired:
            # simulation exceeded timeout; mark as error
            info = 1
            try:
                with open(err_file, 'a') as f:
                    f.write(f'\n[TIMEOUT] Simulation exceeded {sim_timeout}s timeout.\n')
            except Exception:
                pass
        except FileNotFoundError:
            info = 1
            try:
                with open(err_file, 'a') as f:
                    f.write('[ENV] spectre executable could not be launched (FileNotFoundError).\n')
            except Exception:
                pass
        except Exception as e:
            info = 1
            try:
                with open(err_file, 'a') as f:
                    f.write(f'[EXCEPTION] {type(e).__name__}: {e}\n')
            except Exception:
                pass

        return info

    def _create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        """
        Creates a design netlist and run simulation.

        Parameters:
        -----------
        state (dict): Dictionary of design parameter values.
        dsn_name (str, optional): Custom design name. If None, auto-generated from state.
        verbose (bool): If True, prints design name information.
        
        Returns:
        --------
        state (dict): The input design state.
        specs (dict): Dictionary of post-processed simulation results.
        info (int): Error code from simulation (0 = success).
        """
        # generate design name if not provided
        if dsn_name is None:
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

            # post-process results
            if self.post_process:
                try:
                    specs = self.post_process(results, state, self.size_map)
                except TypeError:
                    # Backward compatibility for legacy 2-arg post-process handlers.
                    specs = self.post_process(results, state)
                return state, specs, info
            specs = results

            return state, specs, info
        finally:
            self._cleanup(design_folder, sim_info=info)

    def _cleanup(self, design_folder, sim_info=0):
        """
        Cleans up generated design folder artifacts.
        Behavior controlled by tb_params['keep_raw_artifacts'] flag (Titan-Killer #2).

        Parameters:
        -----------
        design_folder (str): Path to the folder containing the generated netlist and simulation results.
        sim_info (int): Simulation status code (0 = success, non-zero = failure). [Unused with binary flag]
        """
        if os.path.exists(design_folder):
            moved = False
            
            # False: DELETE all artifacts immediately for production mode
            # True: SAVE all artifacts to raw folder for debugging mode
            keep_raw_flag = self.tb_params.get('keep_raw_artifacts', True)
            
            if keep_raw_flag:
                # Debug mode: save all artifacts to raw folder
                try:
                    results_dir = os.environ.get('ASPECTOR_RESULTS_DIR')
                    if results_dir:
                        raw_store = os.path.join(results_dir, "raw")
                        os.makedirs(raw_store, exist_ok=True)
                        dest = os.path.join(raw_store, os.path.basename(design_folder))
                        # avoid clobbering an existing folder
                        if os.path.exists(dest):
                            dest = dest + "_" + uuid.uuid4().hex[:8]
                        shutil.move(design_folder, dest)
                        moved = True
                except Exception as e:
                    try:
                        print(f"[!] Failed to move design to raw folder (debug mode): {e}")
                    except Exception:
                        pass
            
            # If not moved (either production mode or move failed), delete immediately
            if not moved:
                try:
                    shutil.rmtree(design_folder, ignore_errors=False)
                except Exception as e:
                    # fallback to stronger delete
                    subprocess.call(['rm', '-rf', design_folder])

        # remove empty generation directory
        try:
            if os.path.exists(self.gen_dir) and not os.listdir(self.gen_dir):
                os.rmdir(self.gen_dir)
        except OSError:
            pass

    def _parse_result(self, design_folder):
        """
        Parse simulation results from Spectre output files.

        Parameters:
        -----------
        design_folder (str): Path to the design folder containing simulation results.
        
        Returns:
        --------
        res (dict): Dictionary of parsed simulation results.
        """
        # locate raw simulation output
        _, folder_name = os.path.split(design_folder)
        raw_folder = os.path.join(design_folder, '{}.raw'.format(folder_name))

        # parse results
        res = SpectreParser.parse(raw_folder)       

        return res

    def run(self, states, design_names=None, verbose=False):
        """
        Execute simulations for multiple design states.

        Parallelism is managed by runner.py via mp.Pool (Titan-Killer #3).
        This method executes serially; do not use ThreadPool (nested concurrency issue).

        Parameters:
        -----------
        states (list): List of design state dictionaries to simulate.
        design_names (list, optional): Custom design names for each state. If None, names are auto-generated.
        verbose (bool): If True, prints design name information during execution.
        
        Returns:
        --------
        specs (list): List of (state, specs, info) tuples for each design simulated.
        """
        # Serial execution: parallelism is handled by runner.py's mp.Pool (flat hierarchy)
        if design_names is None:
            design_names = [None] * len(states)

        specs = []
        for state, dsn_name in zip(states, design_names):
            result = self._create_design_and_simulate(state, dsn_name, verbose)
            specs.append(result)

        return specs


# ===== Circuit Evaluation Engine ===== #


class EvaluationEngine:
    """
    Main evaluation engine for circuit simulation.

    Initialization Parameters:
    --------------------------
    config (dict): Configuration dictionary specifying measurement and testbench setup.
    """

    def __init__(self, config):

        # accept a configuration dictionary
        self.design_specs = config
        if isinstance(config, dict):
            self.ver_specs = config
        else:
            raise RuntimeError("EvaluationEngine requires a configuration dictionary. YAML files are deprecated.")

        # set up testbench modules
        self.measurement_specs = self.ver_specs['measurement']
        tbs = self.measurement_specs['testbenches']
        self.netlist_module_dict = {}
        for tb_kw, tb_val in tbs.items():
            self.netlist_module_dict[tb_kw] = SpectreWrapper(tb_val)

    def evaluate(self, design_list, debug=True, parallel_config=None):
        """
        Evaluate designs and return processed results.

        Parameters:
        -----------
        design_list (list of dicts): List of design parameter dictionaries.
        debug (bool): If True, prints debug information during evaluation.
        parallel_config (dict, optional): Configuration for parallel execution.
        
        Returns:
        --------
        results (list): List of (state, specs, info) tuples.
        """
        results = []
        
        for state in design_list:
            # run simulations for all testbenches
            sim_results = {}
            for netlist_name, netlist_module in self.netlist_module_dict.items():
                sim_results[netlist_name] = netlist_module._create_design_and_simulate(state, verbose=debug)

            # get specifications from subclass logic when available
            if hasattr(self, 'get_specs'):
                specs_dict = self.get_specs(sim_results, state)
            else:
                # fallback: return first testbench specs
                first_res = list(sim_results.values())[0]
                specs_dict = first_res[1]

            # collect info codes from each netlist simulation
            info_codes = []
            for net_res in sim_results.values():
                try:
                    if isinstance(net_res, tuple) and len(net_res) > 2:
                        info_codes.append(int(net_res[2]))
                except Exception:
                    pass

            # propagate failure if any testbench reported non-zero info
            overall_info = 0
            if any(ic for ic in info_codes):
                overall_info = 1

            results.append((state, specs_dict, overall_info))

        return results

