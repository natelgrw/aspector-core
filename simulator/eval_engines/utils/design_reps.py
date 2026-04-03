"""
design_reps.py

Authors: natelgrw, dkochar
Last Edited: 04/01/2026

Utilities for extracting structured sizing maps from Spectre netlists.
"""

import re

def extract_sizing_map(netlist_path):
    """
    Parse a Spectre netlist and build a component -> optimizable-param map.

    This parser only keeps assignments whose RHS maps to a declared, optimizable
    netlist parameter. That avoids pulling in stale testbench constants and
    directives from older heuristics.

    Parameters:
    -----------
    netlist_path (str): Path to the Spectre netlist file.
    
    Returns:
    --------
    dict: Mapping of component instance names to dicts of optimizable parameters.
    """
    mapping = {}

    non_optimizable_params = {
        "dc_offset", "is_hp", "n_state", "p_state",
        "run_gatekeeper", "run_full_char", "loop_mode", "fet_num", "vdd", "vcm", "tempc", "cload_val"
    }

    # building logical lines by joining Spectre continuation lines ending with '\\'.
    logical_lines = []
    cont = ""
    with open(netlist_path, 'r') as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                if cont:
                    logical_lines.append(cont.strip())
                    cont = ""
                continue

            if stripped.startswith('*') or stripped.startswith('//'):
                continue

            if '//' in stripped:
                stripped = stripped.split('//', 1)[0].rstrip()
                if not stripped:
                    continue

            if stripped.endswith('\\'):
                cont += stripped[:-1].rstrip() + " "
            else:
                logical = (cont + stripped).strip() if cont else stripped
                logical_lines.append(logical)
                cont = ""

    if cont:
        logical_lines.append(cont.strip())

    declared_params = set()
    for line in logical_lines:
        if line.startswith("parameters"):
            declared_params.update(re.findall(r'(\w+)\s*=', line))

    optimizable_params = {
        p for p in declared_params
        if p not in non_optimizable_params and not p.startswith("sim_")
    }

    for line in logical_lines:
        parts = line.split()
        if not parts:
            continue

        name = parts[0]
        if '=' in name:
            continue

        # Component instances only; directives (simulatorOptions, *_sim, etc.) are excluded.
        if not name or name[0].upper() not in {'M', 'R', 'C'}:
            continue

        comp_map = {}
        for token in parts[1:]:
            if '=' not in token:
                continue

            key, val = token.split('=', 1)
            key = key.strip()
            rhs = val.strip()

            if rhs.startswith('{{') and rhs.endswith('}}'):
                rhs = rhs[2:-2].strip()

            rhs = rhs.strip().strip('()').strip()

            if rhs in optimizable_params:
                comp_map[key] = rhs

        if comp_map:
            mapping[name] = comp_map

    return mapping
