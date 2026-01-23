#!/bin/bash

#############################################
# run_optimizer.sh                          #
#                                           #
# Authors: dkochar, natelgrw                #
# Last Edited: 01/15/2026                   #
#                                           #
# Script to run the TURBO optimizer for     #
# hyperparameter tuning.                    #
#############################################

export PYTHONPATH="${PYTHONPATH}:/homes/natelgrw/Documents/titan_foundation_model/turbo_optimizer/working_current"
export BASE_TMP_DIR="/homes/natelgrw/Documents/titan_foundation_model/results"
	source /homes/natelgrw/.bashrc
	conda activate rlenv38
	python /homes/natelgrw/Documents/titan_foundation_model/turbo_optimizer/working_current/sample/random_sample_turbo.py
	conda deactivate
