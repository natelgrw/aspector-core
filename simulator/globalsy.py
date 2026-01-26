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

# netlist target specifications and default weights
# Format: "spec_name": (target_value, weight, simulation_type)
# simulation_type: 0=AC, 1=DC, 2=Noise, 3=Tran
spec_metadata = {
    "gain":             (1.0e5,  50.0, 0),
    "ugbw":             (1.0e9,  10.0, 0),
    "pm":               (60.0,   10.0, 0),
    "cmrr":             (80.0,   10.0, 0),
    "vos":              (1e-3,   50.0, 1),
    "power":            (1.0e-3, 1.0,  1), # Check if 1mW is reasonable, user can override
    "linearity":        (1.0,    5.0,  1), # Assuming unitless or %? Need to verify metric
    "output_voltage_swing":     (0.5,    5.0,  1), # V
    "integrated_noise": (1e-6,   10.0, 2), # Vrms
    "slew_rate":        (10e6,   5.0,  3), # V/s
    "settle_time":    (100e-9, 5.0,  3), # s
}

# Legacy support - will be constructed dynamically in CLI but kept for reference
specs_dict = {k: v[0] for k, v in spec_metadata.items()}
specs_weights = {k: v[1] for k, v in spec_metadata.items()}


# parameter bounds
shared_ranges = {
    'nA': (10e-9, 30e-9),
    'nB': (1, 20),
    'vbiasp': (0, 0.80),
    'vbiasn': (0, 0.80),
    'rr': (5e3, 1e7),
    'cc': (0.1e-12, 2.5e-12)
}
