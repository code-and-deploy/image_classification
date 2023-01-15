EXP_GROUPS = {}

EXP_GROUPS["baselines"] = []
# enumerate different hyperparameters - greedy search is usually desirable
for lr in [1e-1]:
    # for opt in ['adam', 'sgd']:
        EXP_GROUPS["baselines"] += [{"dataset": "digits", "model": "linear", "lr": lr, "opt":'adam', "epochs":10}]
