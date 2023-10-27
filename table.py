import itertools

# Define the hyperparameters and their levels
hyperparameters = [
    "Batch Size",
    "N",
    "SCORE_THRESHOLD",
    "gamma",
    "alpha",
    "beta",
    "gae_lambda",
    "policy_clip",
    "n_epochs",
]

levels = ["Low", "High"]

# Generate all possible combinations of "low" and "high" for the hyperparameters
combinations = list(itertools.product(levels, repeat=len(hyperparameters)))

# Create and print the table
print("Experiment |", " | ".join(hyperparameters))
print("-" * 68)
for i, combo in enumerate(combinations, start=1):
    print(f"{i:<11} |", " | ".join(combo))
