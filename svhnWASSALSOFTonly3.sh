#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="random AL AL_WITHSOFT WASSAL"
SKIP_METHODS="WASSAL_WITHSOFT"
SKIP_BUDGETS="25 50 75 100"
DEVICE_ID="6"
EXPERIMENT_NAME="onlywassal"
SOFT_LOSS_HYPERPARAM="1"
# Call the Python script with the defined arguments
python3 -u tutorials/All_Wassal/wassal_svhn_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/onlywassal/svhnwassalsoft3_may16.log
python3 informme.py "ssw3"