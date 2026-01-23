"""
config_env.py

Author: natelgrw
Last Edited: 01/15/2026

Environment configuration manager for setting up Spectre simulation parameters,
specification ranges, and YAML configuration files for optimization workflows.
"""

import yaml
import os


# ===== Environment Configuration Manager ===== #


class EnvironmentConfig(object):
    """
    Configuration manager for setting up optimization environments.

    Handles the creation and management of simulation parameters, specification
    ranges, and YAML configuration files for Spectre-based circuit optimization.

    Initialization Parameters:
    --------------------------
    netlist_path : str
        Path to the circuit netlist file.
    type : str
        Type of measurement (e.g., 'differential', 'single_ended').
    specs : dict
        Target specifications dictionary.
    params : list
        Parameter names to optimize.
    param_lbs : list
        Lower bounds for parameters.
    param_ubs : list
        Upper bounds for parameters.
    """

    def __init__(self, netlist_path, type, specs, params, param_lbs, param_ubs):

        self.netlist_path = netlist_path
        self.type = type
        self.specs = specs
        self.params = params
        self.param_lbs = param_lbs
        self.param_ubs = param_ubs
        
        # configuration dictionary with default structure
        self.configs = {
            "database_dir": "/homes/natelgrw/Documents/titan_foundation_model/results",
            "measurement": {
                "meas_params": {},
                "testbenches": {
                    "ac_dc": {
                        "netlist_template": self.netlist_path,
                        "tb_module": f"eval_engines.spectre.script_test.{self.type}_meas_man",
                        "tb_class": "ACTB",
                        "post_process_function": "process_ac",
                        "tb_params": {}
                    }
                }
            },
            "params": {},
            "spec_range": {},
            "normalize": {},
            "target_specs": {}
        }
        
        # initialize tracking lists and dictionaries
        self.spec_ranges = []
        self.normalized_list = []
        self.param_dict = {}
        self.yaml_path = ""

    def build_specs(self):
        """
        Build specification ranges and normalized values.

        Processes the target specifications dictionary and creates spec ranges
        for optimization. Phase Margin (PM) gets fixed range around target,
        while other specs get logarithmic ranges.

        Returns:
        --------
        None
            Modifies self.spec_ranges and self.normalized_list in place.
        """
        # iterate through all target specifications
        for spec, val in self.specs.items():
            val = float(val)
            self.configs["target_specs"][spec] = (val,)
        
            if spec == "PM":
                self.spec_ranges.append((val - 30, val + 30, 1))
                self.normalized_list.append(1)
            else:
                self.spec_ranges.append((val / 10, val * 10, val / 100))
                self.normalized_list.append(val / 100)

    def build_params(self):
        """
        Build parameter bounds dictionary with step sizes.

        Creates a mapping of parameter names to (lower_bound, upper_bound, step_size)
        tuples. Step sizes are determined based on parameter type:
        - Biasing and supply parameters: step = 1
        - Current-type parameters (nA*): step = lb/10
        - Bias voltage parameters: step = 0.01
        - Resistance/Capacitance parameters: step = lb/100

        Returns:
        --------
        None
            Modifies self.param_dict in place.
        """
        for i in range(len(self.params)):
            param = self.params[i]
            lb = float(self.param_lbs[i])
            ub = float(self.param_ubs[i])
            
            # supply voltages, common-mode, temperature: fixed step of 1
            if self.params[i] in ["vdd", "vcm", "tempc"] or self.params[i].startswith("nB"):
                self.param_dict[param] = (lb, ub, 1)
            # current parameters: step = 10% of lower bound
            elif self.params[i].startswith("nA"):
                self.param_dict[param] = (lb, ub, lb / 10)
            # bias voltage parameters: fixed small step
            elif self.params[i].startswith("vbias"):
                self.param_dict[param] = (lb, ub, 0.01)
            # resistance and capacitance parameters: step = 1% of lower bound
            elif self.params[i].startswith(("nR", "nC")):
                self.param_dict[param] = (lb, ub, lb / 100)

    def build_configs(self):
        """
        Build complete configuration dictionary.

        Orchestrates the building of specs and params, then populates the
        configuration dictionary with all processed values.

        Returns:
        --------
        None
            Modifies self.configs in place with all parameter and spec information.
        """
        self.build_specs()
        self.build_params()

        self.configs["params"] = self.param_dict
        self.configs["spec_range"] = self.spec_ranges
        self.configs["normalize"] = self.normalized_list

    def write_yaml_configs(self):
        """
        Build and write configuration to YAML file.

        Generates the complete configuration dictionary and writes it to a YAML file
        in the same directory as this Python file. The YAML filename is derived from
        the input netlist filename.

        Returns:
        --------
        str
            Absolute path to the generated YAML configuration file.
        """
        self.build_configs()

        netlist_name = os.path.splitext(os.path.basename(self.netlist_path))[0]
        yaml_filename = f"{netlist_name}.yaml"

        self.yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_filename)

        with open(self.yaml_path, "w") as f:
            yaml.dump(self.configs, f, default_flow_style=False, sort_keys=False)

        print(f"YAML configuration written to {self.yaml_path}")
        return self.yaml_path
    
    def del_yaml(self):
        """
        Delete the generated YAML configuration file.

        Removes the YAML file from disk if it exists. Prints status messages
        indicating success or if the file does not exist.

        Returns:
        --------
        None
        """
        # check if YAML file exists and delete if present
        if os.path.exists(self.yaml_path):
            os.remove(self.yaml_path)
            print(f"Deleted YAML file: {self.yaml_path}")
        else:
            print(f"YAML file does not exist: {self.yaml_path}")
    
