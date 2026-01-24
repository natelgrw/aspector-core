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

# netlist target specifications
specs_dict = {
    "gain": 1.0e5,
    "UGBW": 1.0e9,
    "PM": 60.0,
    "power": 1.0e-6,
}

# parameter bounds
shared_ranges = {
    'nA': (10e-9, 30e-9),
    'nB': (1, 20),
    'vbiasp': (0, 0.80),
    'vbiasn': (0, 0.80),
    'rr': (5e3, 1e7),
    'cc': (0.1e-12, 2.5e-12)
}
