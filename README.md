# ASPECTOR Core

TITAN is a foundation model for the topology and sizing optimization of operational amplifiers, currently under active development.

Current Version: **1.1.0**

## ⚡ Current Features

The TITAN data collection pipeline uses a TuRBO algorithm to optimize sizing, bias, and environment parameters of op-amp Spectre netlists according to 4 performance specs:  

- Gain  
- Unity-Gain Bandwidth (UGBW)  
- Phase Margin (PM)  
- Power  

While simulations compute 11 performance specs in addition to operating point data:  

- Gain  
- Unity-Gain Bandwidth (UGBW)
- Phase Margin (PM)  
- Power  
- Common-Mode Rejection Ratio (CMRR)  
- Input Offset Voltage (Vos)  
- Output Voltage Swing  
- Gain Linearity  
- Integrated Noise  
- Slew Rate  
- Settle Time  

When running the pipeline, each netlist is converted to a `.json` graph before subsequent simulations produce data that is extracted and converted into sizing, operating point, and performance spec `.json` objects. 

## 📖 How to Use

**Requirements:** 
- Linux/Unix environment
- Cadence Spectre, Virtuoso, and ViVA software
- Conda environment with all dependencies specified in `rlenv38.yml`

**1. Clone Repository:**

  ```
  git clone https://github.com/natelgrw/titan_foundation_model.git
  cd titan_foundation_model
  ```

**2. Update Absolute Paths:**
   
   The pipeline currently uses absolute paths in `demo_netlists/` and `turbo_optimizer/`. Modify paths in the relevant scripts/config files to match your local directories.
   
**3. Create & Activate Conda Environment:**

  ```
  conda env create -f rlenv38.yml
  conda activate rlenv38
  ```

**4. Run Optimization Pipeline:**

  ```
  cd turbo_optimizer
  chmod +x run_optimizer.sh
  ./run_optimizer.sh
  ```
  
  This script will prompt you toselect a demo netlist for simulation, run the TuRBO optimization on the specified specs (gain, UGBW, PM, power), perform repeated Spectre simulations to compute all 11 performance metrics, and save optimized topology, sizing, and metrics to the results/ folder.
  
  Ensure all absolute paths are updated before running the pipeline. For multiple netlists, repeat ./run_optimizer.sh or modify the config to include additional files. The pipeline assumes Linux-style paths compatible with Cadence simulations.