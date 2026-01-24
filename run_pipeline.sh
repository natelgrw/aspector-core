#!/bin/bash

#################################################
# run_pipeline.sh                               #
#                                               #
# Authors: dkochar, natelgrw                    #
# Last Edited: 01/24/2026                       #
#                                               #
# Script to run the TURBO optimizer pipeline    #
# for hyperparameter tuning.                    #
#################################################

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"
export BASE_TMP_DIR="${SCRIPT_DIR}/results"
	source ~/.bashrc
	
    # Setup Cadence Environment
    source /usr/share/Modules/init/bash
    module use /opt/mtl-cad/modules
    module load IC
    module load SPECTRE

	conda activate rlenv38
	python "${SCRIPT_DIR}/simulator/cli/cli.py"
	conda deactivate
