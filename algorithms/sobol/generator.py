"""
generator.py

Author: natelgrw
Last Edited: 02/05/2026

Sobol sequence generator for circuit sizing parameters.
Generates valid design points from local technology/config constants.
"""

import math

import numpy as np
from scipy.stats import qmc


# ===== Constants ===== #


TECH_CONSTANTS = {
    7:  {'lmin': 1e-8, 'lmax': 3e-8, 'vdd_nom': 0.70},
    10: {'lmin': 1e-8, 'lmax': 3e-8, 'vdd_nom': 0.75},
    14: {'lmin': 1e-8, 'lmax': 3e-8, 'vdd_nom': 0.80},
    16: {'lmin': 1e-8, 'lmax': 3e-8, 'vdd_nom': 0.80},
    20: {'lmin': 1e-8, 'lmax': 2.4e-8, 'vdd_nom': 0.90},
    'is_hp': [0, 1],
    'process_corner': [(1, 1), (0, 0), (-1, -1), (1, -1), (-1, 1)],
    'shared': {
        'e_series': {
            'E12': [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
            'E24': [
                1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
            ],
        },
        'circuit_params': {
            'Nfin': {'lb': 4, 'ub': 128},
            'vbiasn0': {'lb': 0.45, 'ub': 0.70},
            'vbiasn1': {'lb': 0.45, 'ub': 0.70},
            'vbiasn2': {'lb': 0.65, 'ub': 0.85},
            'vbiasp0': {'lb': 0.40, 'ub': 0.85},
            'vbiasp1': {'lb': 0.40, 'ub': 0.85},
            'vbiasp2': {'lb': 0.15, 'ub': 0.50},
            'nC': {'lb': 100e-15, 'ub': 5e-12, 'series': 'E12'},
            'nR': {'lb': 500, 'ub': 500e3, 'series': 'E24'},
            'vcm': {
                'nmos': (0.65, 0.85),
                'pmos': (0.15, 0.35),
                'default': (0.10, 0.90),
            },
        },
        'testbench_params': {
            'Fet_num': [7, 10, 14, 16, 20],
            'Tempc': {'lb': -40, 'ub': 125},
            'Cload_val': {'lb': 10e-15, 'ub': 5e-12},
        },
    },
}


# ===== Sampling Functions ===== #


def e_series_grid(lb, ub, series='E24'):
    """
    Generate a sorted list of standard E-series values between lb and ub.

    Parameters:
    -----------
    lb (float): Lower bound of the range.
    ub (float): Upper bound of the range.
    series (str): 'E12' or 'E24' for the desired E-series.

    Returns:
    --------
    np.array: Sorted array of E-series values within the specified range.
    """
    e_series = TECH_CONSTANTS['shared']['e_series']
    bases = e_series.get(series, e_series['E12'])
    if lb <= 0:
        lb = 1e-15
    min_exp = int(math.floor(math.log10(lb)))
    max_exp = int(math.ceil(math.log10(ub)))
    vals = []
    for e in range(min_exp - 1, max_exp + 1):
        for b in bases:
            v = b * (10 ** e)
            if v >= lb - 1e-30 and v <= ub + 1e-30:
                vals.append(v)
    vals = sorted(set(vals))
    return np.array(vals)


def discrete_linear_grid(lb, ub, step):
    """
    Generates a discrete linear grid from lb to ub with given step.

    Parameters:
    -----------
    lb (float): Lower bound.
    ub (float): Upper bound.
    step (float): Step size.

    Returns:
    --------
    np.array: Array of discrete values.
    """
    if step <= 0:
        return np.array([lb])
    # ensure inclusion of upper bound
    count = int(math.floor((ub - lb) / step))
    grid = lb + np.arange(0, count + 1) * step
    if grid.size == 0:
        return np.array([lb])
    # ensure ub included
    if grid[-1] < ub - 1e-12:
        grid = np.append(grid, ub)
    return grid


def pick_from_grid_by_u(grid, u):
    """
    Picks a value from grid based on normalized u in [0,1].

    Parameters:
    -----------
    grid (np.array): Grid of values.
    u (float): Normalized value in [0,1].

    Returns:
    --------
    tuple: (value, index)
    """
    if grid is None or len(grid) == 0:
        return None, None
    # map u in [0,1] to an index in [0, len(grid)-1]
    try:
        u_f = float(u)
    except Exception:
        u_f = 0.0
    if u_f < 0.0:
        u_f = 0.0
    if u_f > 1.0:
        u_f = 1.0
    idx = int(u_f * len(grid))
    if idx >= len(grid):
        idx = len(grid) - 1
    return float(grid[idx]), idx


def nearest_grid_u(grid, val):
    """
    Given a grid and a value, returns the normalized u in [0,1] corresponding
    to the nearest grid point.

    Parameters:
    -----------
    grid (np.array): Grid of values.
    val (float): Value to find nearest grid point for.

    Returns:
    --------
    float: Normalized u in [0,1] corresponding to nearest grid point.
    """
    if grid is None or len(grid) == 0:
        return 0.5
    idx = int(np.argmin(np.abs(grid - float(val))))
    if len(grid) <= 1:
        return 0.0
    return (float(idx) + 0.5) / len(grid)


def _read_model_l_bounds(fet_num):
    """
    Reads lmin/lmax directly from TECH_CONSTANTS with safe defaults.

    Parameters:
    -----------
    fet_num (int): Technology node (e.g., 7, 10, 14, 16, 20).

    Returns:
    --------
    tuple: (lmin, lmax) in meters.
    """
    t = TECH_CONSTANTS.get(fet_num, None)
    if t is not None:
        lmin = t.get('lmin', 10e-9)
        lmax = t.get('lmax', max(3 * lmin, lmin + 2e-9))
    else:
        lmin = 10e-9
        lmax = max(3 * lmin, lmin + 2e-9)
    return lmin, lmax


def _vcm_bounds_for_topology(topology_name, vdd_nominal):
    """
    Return topology-aware absolute VCM bounds in volts.

    Parameters:
    -----------
    topology_name (str): Name of the circuit topology (e.g., "nmos_inverter", "pmos_inverter").
    vdd_nominal (float): Nominal supply voltage in volts.

    Returns:
    --------
    tuple: (lb, ub) absolute VCM bounds in volts.   
    """
    fracs = TECH_CONSTANTS['shared']['circuit_params']['vcm']
    try:
        name = (topology_name or '').lower()
    except Exception:
        name = ''

    if 'nmos' in name:
        lb_frac, ub_frac = fracs['nmos']
    elif 'pmos' in name:
        lb_frac, ub_frac = fracs['pmos']
    else:
        lb_frac, ub_frac = fracs['default']
    return lb_frac * vdd_nominal, ub_frac * vdd_nominal


# ===== Sobol Generator Class ===== #


class SobolSizingGenerator:

    def __init__(self, sizing_params_list, seed=None, topology=None):
        """
        Initialize Sobol generator state for sizing + environment dimensions.

        Parameters:
        -----------
        sizing_params_list (list): Parameter names to sample for circuit sizing.
        seed (int | None): Optional Sobol scramble seed.
        topology (str | None): Optional topology name used for VCM fallback bounds.
        """
        self.sizing_params = sizing_params_list
        self.seed = seed
        self.shared_cfg = TECH_CONSTANTS['shared']
        self.circuit_params = self.shared_cfg['circuit_params']
        self.testbench_params = self.shared_cfg['testbench_params']
        self.is_hp_choices = TECH_CONSTANTS['is_hp']
        self.process_corner_choices = TECH_CONSTANTS['process_corner']
        self.tech_nodes = self.testbench_params['Fet_num']
        self.topology = topology

        # fixed testbench parameters
        self.fixed_params = [
            'is_hp', 'process_corner', 'fet_num', 'vdd', 'vcm', 'tempc', 'cload_val'
        ]

        # total dimensions = fixed params + sizing params
        self.dim = len(self.fixed_params) + len(self.sizing_params)
        self.dim_sizing = len(self.sizing_params)

        # initialize Sobol engine
        self.engine = qmc.Sobol(d=self.dim, scramble=True, seed=seed)

    def _draw_sobol(self, n_samples):
        """
        Draw Sobol points robustly across fresh and resumed states.

        Uses random_base2 on power-of-two batch sizes when possible, and
        falls back to random(n) if SciPy rejects base2 due to engine state.
        """
        if n_samples is None or n_samples <= 0:
            return np.empty((0, self.dim))

        m = math.ceil(math.log2(n_samples)) if n_samples > 0 else 0
        if m <= 0:
            return self.engine.random(n=n_samples)

        try:
            u_full = self.engine.random_base2(m)
            return u_full[:n_samples]
        except ValueError:
            # Resumed/non-power-of-two engine state: fall back to unrestricted draw.
            return self.engine.random(n=n_samples)

    def _log_sample(self, u, lb, ub):
        """
        Sample logarithmically between lower and upper bounds.

        Parameters:
        -----------
        u (float): Normalized coordinate in [0, 1].
        lb (float): Lower bound (> 0).
        ub (float): Upper bound (> 0).

        Returns:
        --------
        float: Log-space sampled value.
        """
        log_lb = np.log10(lb)
        log_ub = np.log10(ub)
        val_log = log_lb + float(u) * (log_ub - log_lb)
        return 10 ** val_log

    def _pick_tb_val(self, param_key, series_key, u_val):
        """
        Pick a testbench value from E-series grid or log fallback.

        Parameters:
        -----------
        param_key (str): Key under local testbench parameter config.
        series_key (str | None): Key under local sampling metadata (or explicit series).
        u_val (float): Normalized coordinate in [0, 1].

        Returns:
        --------
        float | None: Sampled value, or None if bounds are unavailable.
        """
        try:
            pinfo = self.testbench_params[param_key]
            lb = pinfo['lb']
            ub = pinfo['ub']
        except Exception:
            return None
        if series_key is None:
            return self._log_sample(u_val, lb, ub)
        grid = e_series_grid(lb, ub, series=series_key)
        val, _ = pick_from_grid_by_u(grid, u_val)
        if val is None:
            val = self._log_sample(u_val, lb, ub)
        return val

    def _resolve_bias_bounds(self, param_name, vdd_selected):
        """
        Resolve vbias bounds for a parameter using selected/sample VDD.

        Parameters:
        -----------
        param_name (str): Bias parameter name (e.g., vbiasn0, vbiasp1).
        vdd_selected (float): Selected/sample VDD for this design point.

        Returns:
        --------
        tuple: (lb, ub) as floats, or (None, None) when unavailable.
        """
        if not isinstance(param_name, str):
            return None, None

        if param_name.startswith('vbiasn'):
            suffix = param_name[len('vbiasn'):]
            key = 'vbiasn' + (suffix if suffix else '0')
        elif param_name.startswith('vbiasp'):
            suffix = param_name[len('vbiasp'):]
            key = 'vbiasp' + (suffix if suffix else '0')
        else:
            return None, None

        binfo = self.circuit_params.get(key, None)
        if not isinstance(binfo, dict):
            return None, None

        def _to_fraction(expr):
            try:
                if isinstance(expr, str):
                    return float(
                        eval(
                            expr,
                            {"__builtins__": None},
                            {
                                # Legacy compatibility: interpret expressions as fractions.
                                "vdd_nominal": 1.0,
                                "vdd_selected": 1.0,
                            },
                        )
                    )
                return float(expr)
            except Exception:
                return None

        lb_frac = _to_fraction(binfo.get('lb'))
        ub_frac = _to_fraction(binfo.get('ub'))
        if lb_frac is None or ub_frac is None:
            return None, None

        vdd_val = float(vdd_selected)
        lb = lb_frac * vdd_val
        ub = ub_frac * vdd_val
        if ub < lb:
            ub = lb
        return lb, ub

    def _pick_temp_from_u(self, u_val):
        """
        Map normalized value to integer temperature bounds.

        Parameters:
        -----------
        u_val (float): Normalized coordinate in [0, 1].

        Returns:
        --------
        int: Temperature in degrees C, clamped to configured bounds.
        """
        temp_lb = int(round(self.testbench_params['Tempc']['lb']))
        temp_ub = int(round(self.testbench_params['Tempc']['ub']))
        t = int(round(temp_lb + float(u_val) * (temp_ub - temp_lb)))
        return max(temp_lb, min(temp_ub, t))

    def sample_context_u(self, n_samples):
        """
        Draw Sobol context vectors for environment dimensions only.

        Parameters:
        -----------
        n_samples (int): Number of context vectors to draw.

        Returns:
        --------
        np.ndarray: Shape (n_samples, len(self.fixed_params)) in [0, 1].
        """
        if n_samples is None or n_samples <= 0:
            return np.empty((0, len(self.fixed_params)))

        u_ctx = self._draw_sobol(n_samples)[:, :len(self.fixed_params)]
        return u_ctx

    def generate(self, n_samples=None, u_samples=None, robust_env=False, start_idx=0):
        """
        Generates n_samples of valid design configurations with environment as context inputs.

        Parameters:
        -----------
        n_samples (int): Number of samples to generate.
        u_samples (np.array): Pre-generated [0,1] samples of shape (n, dim) or (n, dim_sizing).
        robust_env (bool): Deprecated compatibility flag. Environment values come
            directly from full-dimension inputs, or are Sobol-sampled when absent.
        start_idx (int): Index to resume Sobol sequence from.

        Returns:
        --------
        list[dict]: List of dictionaries with both sizing and environment parameters.
        """
        # expand sizing-only inputs with environment dimensions when needed
        if u_samples is not None:
            n_samples = len(u_samples)
            input_dim = len(u_samples[0]) if isinstance(u_samples, list) else u_samples.shape[1]

            if input_dim == self.dim:
                u_samples_expanded = u_samples
            elif input_dim == self.dim_sizing:
                # Backward-compatible expansion path for legacy sizing-only callers.
                # Contextual TuRBO now uses full-dimension inputs and bypasses this.
                if n_samples > 0:
                    u_env = self._draw_sobol(n_samples)[:, :len(self.fixed_params)]
                else:
                    u_env = np.empty((0, len(self.fixed_params)))
                u_samples_expanded = np.hstack([u_samples, u_env])
            else:
                raise ValueError(
                    f"Provided samples dim {input_dim} matches neither full ({self.dim}) nor sizing ({self.dim_sizing})"
                )
        else:
            if n_samples is None:
                raise ValueError("Must provide n_samples if u_samples is None")
            if start_idx > 0:
                self.engine.fast_forward(start_idx)
                u_samples_expanded = self._draw_sobol(n_samples)
            else:
                u_samples_expanded = self._draw_sobol(n_samples)
        
        configs = []
        n_env_dims = len(self.fixed_params)
        
        # --- Environment parameter sampling ---
        for i in range(n_samples):
            row = u_samples_expanded[i]
            config = {}
            env_col_idx = 0

            # is_hp
            u_is_hp = row[env_col_idx]
            env_col_idx += 1
            is_hp_idx = int(u_is_hp * len(self.is_hp_choices))
            if is_hp_idx >= len(self.is_hp_choices):
                is_hp_idx = len(self.is_hp_choices) - 1
            is_hp = self.is_hp_choices[is_hp_idx]
            config['is_hp'] = is_hp

            # n_state, p_state
            u_state = row[env_col_idx]
            env_col_idx += 1
            state_idx = int(u_state * len(self.process_corner_choices))
            if state_idx >= len(self.process_corner_choices):
                state_idx = len(self.process_corner_choices) - 1
            n_state, p_state = self.process_corner_choices[state_idx]
            config['n_state'] = n_state
            config['p_state'] = p_state

            # fet_num (tech node)
            u_fet = row[env_col_idx]
            env_col_idx += 1
            fet_idx = int(u_fet * len(self.tech_nodes))
            fet_idx = min(fet_idx, len(self.tech_nodes) - 1)
            fet_num = self.tech_nodes[fet_idx]
            config['fet_num'] = fet_num

            # node-dependent constants
            t_const = TECH_CONSTANTS[fet_num]
            vdd_nom = t_const['vdd_nom']

            # VDD
            u_vdd = row[env_col_idx]
            env_col_idx += 1
            vdd_lb = 0.9 * vdd_nom
            vdd_ub = 1.1 * vdd_nom
            abs_step = 0.01
            try:
                step_vdd = abs_step
                vdd_lb = round(vdd_lb / step_vdd) * step_vdd
                vdd_ub = round(vdd_ub / step_vdd) * step_vdd
                if vdd_ub < vdd_lb:
                    vdd_ub = vdd_lb
            except Exception:
                pass
            vdd_grid = discrete_linear_grid(vdd_lb, vdd_ub, step_vdd)
            vdd, _ = pick_from_grid_by_u(vdd_grid, u_vdd)
            config['vdd'] = vdd

            # VCM
            u_vcm = row[env_col_idx]
            env_col_idx += 1

            # Topology-based VCM bounds only.
            vcm_lb, vcm_ub = _vcm_bounds_for_topology(self.topology, vdd)
            if vcm_ub < vcm_lb:
                vcm_ub = vcm_lb
            abs_step = 0.01
            try:
                step_vcm = abs_step
                vcm_lb = round(vcm_lb / step_vcm) * step_vcm
                vcm_ub = round(vcm_ub / step_vcm) * step_vcm
                if vcm_ub < vcm_lb:
                    vcm_ub = vcm_lb
            except Exception:
                pass
            vcm_grid = discrete_linear_grid(vcm_lb, vcm_ub, step_vcm)
            vcm, _ = pick_from_grid_by_u(vcm_grid, u_vcm)
            config['vcm'] = vcm

            # temperature
            u_temp = row[env_col_idx]
            env_col_idx += 1
            tempc = self._pick_temp_from_u(u_temp)
            config['tempc'] = tempc

            # load capacitance
            u_cl = row[env_col_idx]
            env_col_idx += 1
            config['cload_val'] = self._pick_tb_val('Cload_val', 'C_series', u_cl)
            
            # sizing parameters
            col_idx = n_env_dims
            for param in self.sizing_params:
                u_p = row[col_idx]
                col_idx += 1
                                
                if param.startswith('nA'):
                    model_lmin, model_lmax = _read_model_l_bounds(fet_num)
                    if model_lmax <= model_lmin:
                        model_lmax = model_lmin + 1e-9

                    step = 1e-9
                    grid = np.arange(model_lmin, model_lmax + 1e-15, step)
                    if grid.size == 0:
                        grid = np.array([model_lmin])

                    # map u_p in [0,1] to an index in the discrete grid
                    idx = int(u_p * grid.size)
                    if idx >= grid.size:
                        idx = grid.size - 1
                    val = float(grid[idx])
                    config[param] = val
                    
                elif param.startswith('nB'):
                    bp_nfin = self.circuit_params.get('Nfin', None)
                    p_lb = int(bp_nfin['lb'])
                    p_ub = int(bp_nfin['ub'])
                    # map u_p in [0,1] to integer range [p_lb, p_ub]
                    span = max(0, p_ub - p_lb)
                    val = int(round(p_lb + u_p * span))
                    if val < p_lb:
                        val = p_lb
                    if val > p_ub:
                        val = p_ub
                    config[param] = int(val)
                    
                elif "bias" in param:
                    vdd = config['vdd']

                    # NMOS vbiasn
                    if param.startswith('vbiasn'):
                        p_lb, p_ub = self._resolve_bias_bounds(param, vdd)
                        if p_lb is None or p_ub is None:
                            p_lb, p_ub = 0.0, float(vdd)
                        step_bias = 0.01
                        bias_grid = discrete_linear_grid(p_lb, p_ub, step_bias)
                        val, _ = pick_from_grid_by_u(bias_grid, u_p)
                        if val is None:
                            val = p_lb + u_p * (p_ub - p_lb)
                        # snap to canonical step to ensure exact multiples
                        try:
                            if step_bias > 0:
                                val = p_lb + round((val - p_lb) / step_bias) * step_bias
                        except Exception:
                            pass
                        # clamp
                        if val < p_lb:
                            val = p_lb
                        if val > p_ub:
                            val = p_ub
                        config[param] = float(val)

                    # PMOS vbiasp
                    elif param.startswith('vbiasp'):
                        p_lb, p_ub = self._resolve_bias_bounds(param, vdd)
                        if p_lb is None or p_ub is None:
                            p_lb, p_ub = 0.0, float(vdd)
                        step_bias = 0.01
                        bias_grid = discrete_linear_grid(p_lb, p_ub, step_bias)
                        val, _ = pick_from_grid_by_u(bias_grid, u_p)
                        if val is None:
                            val = p_lb + u_p * (p_ub - p_lb)
                        try:
                            if step_bias > 0:
                                val = p_lb + round((val - p_lb) / step_bias) * step_bias
                        except Exception:
                            pass
                        if val < p_lb:
                            val = p_lb
                        if val > p_ub:
                            val = p_ub
                        config[param] = float(val)

                    else:
                        # generic bias fallback
                        p_lb = 0.1 * config['vdd']
                        p_ub = 0.9 * config['vdd']
                        step_bias = 0.01
                        bias_grid = discrete_linear_grid(p_lb, p_ub, step_bias)
                        val, _ = pick_from_grid_by_u(bias_grid, u_p)
                        if val is None:
                            val = p_lb + u_p * (p_ub - p_lb)
                        config[param] = val

                elif param.startswith('nC'):
                    # discrete capacitor values from circuit_params
                    c_info = self.circuit_params.get('nC', {})
                    p_lb = c_info.get('lb', 100e-15)
                    p_ub = c_info.get('ub', 5e-12)
                    c_series = c_info.get('series', 'E12')
                    c_grid = e_series_grid(p_lb, p_ub, series=c_series)
                    c_val, _ = pick_from_grid_by_u(c_grid, u_p)
                    if c_val is None:
                        c_val = self._log_sample(u_p, p_lb, p_ub)
                    config[param] = c_val

                elif param.startswith('nR'):
                    # discrete resistor values from circuit_params
                    r_info = self.circuit_params.get('nR', {})
                    p_lb = r_info.get('lb', 500)
                    p_ub = r_info.get('ub', 500e3)
                    r_series = r_info.get('series', 'E24')
                    r_grid = e_series_grid(p_lb, p_ub, series=r_series)
                    r_val, _ = pick_from_grid_by_u(r_grid, u_p)
                    if r_val is None:
                        r_val = self._log_sample(u_p, p_lb, p_ub)
                    config[param] = r_val
                    
                else:
                    # fallback for unknown parameters
                    config[param] = u_p

            configs.append(config)
            
        return configs

    def inverse_map(self, df):
        """
        Maps physical environment + sizing parameters back to Unit Hypercube [0,1]^d.
        Used for initializing TuRBO with existing data ("Sight" mode).
        
        Parameters:
        -----------
        df (pd.DataFrame): Dataframe containing 'in_{param}' columns (or without 'in_' prefix).

        Returns:
        --------
        tuple: (X_tensor, valid_idx)
            X_tensor is shape [N, dim] with normalized values in [0,1].
            valid_idx are row indices from the input dataframe that were kept.
        """
        import torch
        
        fet_num = self.tech_nodes[0]
        t_const = TECH_CONSTANTS[fet_num]
        current_vdd = t_const['vdd_nom']
        
        full_param_list = self.fixed_params + self.sizing_params
        X_rows = []
        valid_idx = []
        
        for i_row, (_, row) in enumerate(df.iterrows()):
            u_row = []
            
            try:
                for param in full_param_list:
                    # Handle process corner before generic column lookup.
                    # Input rows carry n_state/p_state fields, not process_corner.
                    if param == 'process_corner':
                        n_val = row.get('in_n_state', row.get('n_state', None))
                        p_val = row.get('in_p_state', row.get('p_state', None))

                        if n_val is not None and p_val is not None:
                            state_tuple = (int(n_val), int(p_val))
                            try:
                                state_idx = self.process_corner_choices.index(state_tuple)
                                u = (float(state_idx) + 0.5) / len(self.process_corner_choices)
                            except ValueError:
                                u = 0.5
                        else:
                            u = 0.5

                        u_row.append(max(0.0, min(1.0, float(u))))
                        continue

                    col_name = f"in_{param}"
                    if col_name not in row:
                        col_name = param
                    
                    if col_name not in row:
                        continue
                    else:
                        val = row[col_name]
                        u = 0.5
                        
                        if param == 'fet_num':
                            try:
                                fet_num = int(val)
                                node_idx = self.tech_nodes.index(fet_num)
                                current_vdd = TECH_CONSTANTS[fet_num]['vdd_nom']
                                u = (float(node_idx) + 0.5) / len(self.tech_nodes)
                            except Exception:
                                u = 0.5

                        elif param == 'is_hp':
                            try:
                                is_hp_idx = self.is_hp_choices.index(int(val))
                                u = (float(is_hp_idx) + 0.5) / len(self.is_hp_choices)
                            except (ValueError, AttributeError):
                                u = 0.5
                            
                        elif param == 'vdd':
                            vdd_nom = current_vdd
                            vdd_lb = 0.9 * vdd_nom
                            vdd_ub = 1.1 * vdd_nom
                            vdd_grid = discrete_linear_grid(
                                round(vdd_lb / 0.01) * 0.01,
                                round(vdd_ub / 0.01) * 0.01,
                                0.01
                            )
                            u = nearest_grid_u(vdd_grid, val)
                        
                        elif param == 'vcm':
                            try:
                                vdd_for_vcm = float(row.get('in_vdd', row.get('vdd', current_vdd)))
                            except Exception:
                                vdd_for_vcm = current_vdd
                            vcm_lb, vcm_ub = _vcm_bounds_for_topology(self.topology, vdd_for_vcm)
                            vcm_grid = discrete_linear_grid(
                                round(vcm_lb / 0.01) * 0.01,
                                round(vcm_ub / 0.01) * 0.01,
                                0.01
                            )
                            u = nearest_grid_u(vcm_grid, val)
                        
                        elif param == 'tempc':
                            temp_lb = int(round(self.testbench_params['Tempc']['lb']))
                            temp_ub = int(round(self.testbench_params['Tempc']['ub']))
                            temp_span = max(1, temp_ub - temp_lb)
                            u = (int(val) - temp_lb) / temp_span
                        
                        elif param == 'cload_val':
                            cl_lb = self.testbench_params['Cload_val']['lb']
                            cl_ub = self.testbench_params['Cload_val']['ub']
                            # Use C_series from circuit_params if available, else fallback to E12
                            c_info = self.circuit_params.get('nC', {})
                            cl_series = c_info.get('series', 'E12')
                            cl_grid = e_series_grid(cl_lb, cl_ub, series=cl_series)
                            if len(cl_grid) > 0:
                                u = nearest_grid_u(cl_grid, val)
                            else:
                                u = (np.log10(val) - np.log10(cl_lb)) / (np.log10(cl_ub) - np.log10(cl_lb))
                        
                        elif param.startswith('nA'):
                            model_lmin, model_lmax = _read_model_l_bounds(fet_num)
                            if model_lmax <= model_lmin:
                                model_lmax = model_lmin + 1e-9
                            grid = np.arange(model_lmin, model_lmax + 1e-15, 1e-9)
                            if grid.size == 0:
                                u = 0.5
                            else:
                                idx_nearest = int(np.argmin(np.abs(grid - float(val))))
                                u = float(idx_nearest) / max(1, (grid.size - 1))
                        
                        elif param.startswith('nB'):
                            bp_nfin = self.circuit_params['Nfin']
                            p_lb = int(bp_nfin['lb'])
                            p_ub = int(bp_nfin['ub'])
                            u = (val - p_lb) / (p_ub - p_lb)
                        
                        elif "bias" in param:
                            try:
                                vdd_for_bias = float(row.get('in_vdd', row.get('vdd', current_vdd)))
                            except Exception:
                                vdd_for_bias = current_vdd

                            if param.startswith('vbiasn'):
                                p_lb, p_ub = self._resolve_bias_bounds(param, vdd_for_bias)
                                if p_lb is None or p_ub is None:
                                    p_lb, p_ub = 0.0, float(vdd_for_bias)
                            elif param.startswith('vbiasp'):
                                p_lb, p_ub = self._resolve_bias_bounds(param, vdd_for_bias)
                                if p_lb is None or p_ub is None:
                                    p_lb, p_ub = 0.0, float(vdd_for_bias)
                            else:
                                p_lb = 0.0
                                p_ub = float(vdd_for_bias)

                            bias_grid = discrete_linear_grid(p_lb, p_ub, 0.01)
                            u = nearest_grid_u(bias_grid, val)
                        
                        elif param.startswith('nC'):
                            c_info = self.circuit_params.get('nC', {})
                            p_lb = c_info.get('lb', 100e-15)
                            p_ub = c_info.get('ub', 5e-12)
                            c_series = c_info.get('series', 'E12')
                            c_grid = e_series_grid(p_lb, p_ub, series=c_series)
                            u = nearest_grid_u(c_grid, val)

                        elif param.startswith('nR'):
                            r_info = self.circuit_params.get('nR', {})
                            p_lb = r_info.get('lb', 500)
                            p_ub = r_info.get('ub', 500e3)
                            r_series = r_info.get('series', 'E24')
                            r_grid = e_series_grid(p_lb, p_ub, series=r_series)
                            u = nearest_grid_u(r_grid, val)
                        
                        else:
                            u = val
                    
                    u = max(0.0, min(1.0, float(u)))
                    u_row.append(u)
                
                if len(u_row) == len(full_param_list):
                    X_rows.append(u_row)
                    valid_idx.append(i_row)
            
            except Exception:
                continue
        
        return torch.tensor(X_rows, dtype=torch.double), valid_idx


# ===== Main ===== #

if __name__ == "__main__":
    test_params = ['nA1', 'nB1', 'vbiasn0', 'nR1']
    gen = SobolSizingGenerator(test_params)
    samples = gen.generate(5)
    
    import json
    print(json.dumps(samples, indent=2))
