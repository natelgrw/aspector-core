"""
globalsy.py

Author: natelgrw
Last Edited: 01/15/2026

Global variables for TURBO optimizer module.
"""

counterrrr = 0

# transistor power states
region_mapping = {
    0: "cut-off",
    1: "triode",
    2: "saturation",
    3: "sub-threshold",
    4: "breakdown"
}

# NOTE: The optimizer will handle spec optimization weights and targets.
# Users should not implement them here manually.
spec_metadata = {}
specs_dict = {}
specs_weights = {}

# Parameter Restrictions

# Basic Parameters
basic_params = {
    'L': {
        'type': 'continuous',
        'lb': '1.1 * L_min', # Dependent on technology node
        'ub': '10 * L_min',
        'unit': 'm'
    },
    'Nfin': {
        'type': 'integer',
        'lb': 1,
        'ub': 256,
        'unit': ''
    },
    'C_internal': {
        'type': 'continuous',
        'lb': 100e-15, # 100 fF
        'ub': 5e-12,   # 5 pF
        'sampling': 'logarithmic'
    },
    'R_internal': {
        'type': 'continuous',
        'lb': 500,     # 500 Ohm
        'ub': 500e3,   # 500 kOhm
        'sampling': 'logarithmic'
    },
    'Ibias': {
        'type': 'continuous',
        'lb': 100e-9,  # 100 nA
        'ub': 1e-3,    # 1 mA
        'unit': 'A'
    },
    'Vbias': {
        'type': 'continuous',
        'lb': 0,
        'ub': 'VDD',   # Dependent on VDD
        'unit': 'V'
    }
}

# Environment Parameters
env_params = {
    'Rfeedback_val': {
        'type': 'continuous',
        'lb': 1e3,     # 1 kOhm
        'ub': 1e6,     # 1 MOhm
        'sampling': 'logarithmic'
    },
    'R_src': {
        'type': 'continuous',
        'lb': 50,      # 50 Ohm
        'ub': 100e3,   # 100 kOhm
        'sampling': 'logarithmic'
    },
    'Cload_val': {
        'type': 'continuous',
        'lb': 10e-15,  # 10 fF
        'ub': 10e-12,  # 10 pF
        'sampling': 'logarithmic'
    }
}

# Testbench Parameters
testbench_params = {
    'Fet_num': [7, 10, 14, 16, 20],
    'VDD': {
        'type': 'continuous',
        'lb': '0.9 * vdd_nominal',
        'ub': '1.1 * vdd_nominal'
    },
    'VCM': {
        'type': 'continuous',
        'lb': 0.15,
        'ub': 'VDD - 0.15'
    },
    'Tempc': {
        'type': 'continuous',
        'lb': -40,
        'ub': 125
    }
}

# Legacy support - shared_ranges
# Note: These legacy ranges are overridden by the structured parameters above
# where applicable, but kept for compatibility with existing extractors
# until they are fully updated.
shared_ranges = {
    'nA': (10e-9, 100e-9),  # Rough default for L
    'nB': (1, 256),         # Nfin
    'vbiasp': (0, 1.2),     # Vbias default
    'vbiasn': (0, 1.2),     # Vbias default
    'rr': (500, 500e3),     # R_internal
    'cc': (100e-15, 5e-12), # C_internal
    'rfeedback': (1e3, 1e6),
    'rsrc': (50, 100e3),
    'cload': (10e-15, 10e-12)
}
