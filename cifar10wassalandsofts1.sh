#!/bin/bash

# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL random WASSAL_WITHSOFT"
SKIP_METHODS="WASSAL_WITHSOFT glister_withsoft gradmatch-tss_withsoft coreset_withsoft"
SKIP_BUDGETS="20 30 40 50 60 70 80 90 100"
DEVICE_ID="0"
EXPERIMENT_NAME="Softloss0.3"
SOFT_LOSS_HYPERPARAM="0.3"
# Call the Python script with the defined arguments
python3 -u tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HPERPARAM" 2>&1 | tee  tutorials/results/cifar10_10rounds_wassal1_small.log
python3 informme.py