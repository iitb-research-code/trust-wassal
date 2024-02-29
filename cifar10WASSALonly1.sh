#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL AL_WITHSOFT WASSAL_WITHSOFT"
SKIP_METHODS="WASSAL"
SKIP_BUDGETS="25 50 100"
DEVICE_ID="3"
EXPERIMENT_NAME="onlywassal"
SOFT_LOSS_HYPERPARAM="0.3"
# Call the Python script with the defined arguments
python3 -u tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/onlywassal/cifar10onlywassal1.log
python3 informme.py