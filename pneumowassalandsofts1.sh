# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL random"
SKIP_METHODS="WASSAL_WITHSOFT glister_withsoft gradmatch-tss_withsoft coreset_withsoft"
SKIP_BUDGETS="20 30 40 50 60 70 80 90 100"
DEVICE_ID="1"
EXPERIMENT_NAME="softloss0.3"
SOFT_LOSS_HYPERPARAM="0.3"
python3 -u tutorials/All_Wassal/wassal_pneumonia_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/softloss0.3/pneumo1.log
python3 informme.py