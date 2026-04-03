"""
config_env.py

Author: natelgrw
Last Edited: 01/15/2026

Environment configuration manager for building runtime simulation config dicts.
"""


# ===== Environment Configuration Manager ===== #


class EnvironmentConfig:

    def __init__(self, netlist_path, type, specs, params, param_lbs, param_ubs, results_dir="results"):
        """
        Initializes the EnvironmentConfig class - a configuration manager for Cadence simulation environments.

        Parameters:
        -----------
        netlist_path (str): Path to the circuit netlist file.
        type (str): Type of measurement (e.g., 'differential', 'single_ended').
        specs (dict): Target specifications dictionary.
        params (list): Parameter names to optimize.
        param_lbs (list): Lower bounds for parameters.
        param_ubs (list): Upper bounds for parameters.
        results_dir (str): Directory to save simulation results.
        """
        self.netlist_path = netlist_path
        self.type = type
        self.specs = specs
        self.params = params
        self.param_lbs = param_lbs
        self.param_ubs = param_ubs
        
        self.configs = {
            "database_dir": results_dir,
            "measurement": {
                "meas_params": {},
                "testbenches": {
                    "ac_dc": {
                        "netlist_template": self.netlist_path,
                        "tb_module": f"simulator.eval_engines.spectre.measurements.{self.type}_meas_man",
                        "tb_class": "ACTB",
                        "post_process_function": "process_ac",
                        "tb_params": {}
                    }
                }
            },
            "target_specs": {}
        }

    def build_specs(self):
        """
        Builds a target specification dictionary.
        Handles both scalar targets and range-based targets (e.g., PM: [60°, 65°]).
        Prevents "Single-Value Target Trap" (Titan-Killer #1 for config).
        """
        for spec, val in self.specs.items():
            # check if val is already a range
            if isinstance(val, (list, tuple)) and len(val) == 2:
                # already a range target; pass through as-is
                try:
                    val_range = (float(val[0]), float(val[1]))
                    self.configs["target_specs"][spec] = val_range
                except (ValueError, TypeError):
                    # fallback: treat as scalar
                    self.configs["target_specs"][spec] = float(val)
            else:
                self.configs["target_specs"][spec] = float(val)

    def build_configs(self):
        """
        Builds a complete configuration dictionary.
        """
        self.build_specs()

    def write_yaml_configs(self):
        """
        Compatibility shim that returns the configuration dictionary.
        """
        self.build_configs()
        return self.get_config_dict()

    def get_config_dict(self):
        """
        Returns the configuration as a Python dictionary.
        """
        self.build_configs()
        return dict(self.configs)
    
    def del_yaml(self):
        """
        No-op: YAML files are deprecated. Kept for backward compatibility.
        """
        return
    
