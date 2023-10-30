# Define the skip_strategies and skip_budgets
SKIP_STRATEGIES="AL random WASSAL"
SKIP_METHODS="WASSAL_WITHSOFT glister_withsoft gradmatch-tss_withsoft coreset_withsoft"
SKIP_BUDGETS="25 50 75 100 200"
DEVICE_ID="0"

python3 tutorials/All_Wassal/wassal_pneumonia_multiclass_vanilla.py "$SKIP_STRATEGIES" "$SKIP_METHODS" "$SKIP_BUDGETS" "$DEVICE_ID" 2>&1 | tee tutorials/results/pneumowassalandsofts1.log
python3 informme.py