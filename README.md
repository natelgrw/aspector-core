# ASPECTOR Core

A Cadence Spectre simulation pipeline for op-amp netlists. Used as part of the ASPECTOR analog design suite to extract performance data for dataset construction.

Current Version: **1.2.0**

## ⚡ Current Features

ASPECTOR Core boasts a sleek, comprehensive terminal CLI to extract simulation data.

A TuRBO algorithm is used to optimize sizing, bias, and environment parameters of op-amp Spectre netlists according to 4 performance specs:  

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
   
**2. Create & Activate Conda Environment:**

  ```
  conda env create -f rlenv38.yml
  conda activate rlenv38
  ```

**3. Run Optimization Pipeline:**

  ```
  cd turbo_optimizer
  chmod +x run_optimizer.sh
  ./run_optimizer.sh
  ```
  
  This script will prompt you toselect a demo netlist for simulation, run the TuRBO optimization on the specified specs (gain, UGBW, PM, power), perform repeated Spectre simulations to compute all 11 performance metrics, and save optimized topology, sizing, and metrics to the results/ folder.
  