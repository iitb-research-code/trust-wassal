#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL_WITHSOFT WASSAL WASSAL_WITHSOFT"
SKIP_METHODS=""
SKIP_BUDGETS=""
DEVICE_ID="2"
EXPERIMENT_NAME="onlyal"
SOFT_LOSS_HYPERPARAM="0.3"
# Call the Python script with the defined arguments
python3 tutorials/All_Wassal/wassal_svhn_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" 2>&1 | tee tutorials/results/onlyal/svhn10onlyal.log
python3 informme.py