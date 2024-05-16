# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="random AL WASSAL AL_WITHSOFT"
SKIP_METHODS="WASSAL_WITHSOFT"
SKIP_BUDGETS="70 80 90 100"
DEVICE_ID="0"
EXPERIMENT_NAME="onlywassal"
SOFT_LOSS_HYPERPARAM="1"
python3 -u tutorials/All_Wassal/wassal_pneumonia_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" "$EXPERIMENT_NAME" "$SOFT_LOSS_HYPERPARAM" 2>&1 | tee tutorials/results/onlywassal/pneumowassalsoft2_expdatedmay16.log
python3 informme.py "psw2"