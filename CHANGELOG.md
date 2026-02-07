Version **1.4.0**

Date Released: **02/06/2026**

- Complete redesign of op-amp netlist testbenches
- Corrected spec measurement and extraction functions
- New search algorithms implemented to explore design space, including Sobol, LHS, and more
- Updated simulation and data extraction pipeline accomodating new testbench

Version **1.3.0**

Date Released: **01/26/2026**

- Customization of specs extracted from simulation
- Customization of optimization specs and weights

Version **1.2.0**

Date Released: **01/24/2026**

- Newly formatted CLI
- Complete reorganization of codebase
- Automated removal of heavy leftover simulation files after data extraction
- Removal of absolute file paths for filesystem configs

Version **1.1.0**

Date Released: **01/22/2026**

- Netlist -> Graph script integration
- Revamped data collection pipeline with formatted `json` output
- Cleaned up codebase

Version **1.0.0**

Date Released: **11/24/2025**

- Full TuRBO-powered Spectre netlist simulation pipeline
- Optimization across 4 parameters
- Calculation of 11 performance specs from sized configurations