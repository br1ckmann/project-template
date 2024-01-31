# LOAD MODULES
# Standard library
import os
import sys
import itertools
import threadpoolctl

# Third party
from tqdm import tqdm

# NOTE: Your script is not in the root directory. We must hence change the system path
DIR = "/Users/cbr/Code/project-template"
os.chdir(DIR)
sys.path.append(DIR)

# Proprietary
from src.data.ihdp_s_1 import load_data
from src.models.neural import MLP
from src.utils.metrics import mean_integrated_prediction_error
from src.utils.setup import (
    load_config,
    check_create_csv,
    get_rows,
    add_row,
    add_dict,
)
from src.utils.training import train_val_tuner

# SETTINGS
# Directory
# NOTE: I am working with a tracker that keeps track of the experiments that have already been completed. 
# This is useful if you want to run the script in parallel on multiple machines. 
# If you do not want to use this, you can simply remove the tracker and the corresponding code.
RES_FILE = "results.csv"
TRACKER = "tracker.csv"

# SETUP
# Load config
CONFIG = load_config("config/data/ihdp_s_1.yaml")["parameters"]
HYPERPARAMS = load_config("config/models/mlp.yaml")["parameters"]

# Number of parameter combinations to consider
RANDOM_SEARCH_N = 3

# Save para combinations
combinations = list(itertools.product(*CONFIG.values()))

# Ini tracker
check_create_csv(TRACKER, CONFIG.keys())

for combination in tqdm(combinations, desc="Iterate over combinations"):
    # Get completed combinations
    completed = get_rows(TRACKER)
    
    # If combination already completed, skip
    if (combination in completed):
        continue

    # Add combination to tracker
    add_row(TRACKER, combination)

    # Save settings
    data_settings = dict(zip(CONFIG.keys(), combination))
    
    # Ini results
    results = {}
    
    # Log settings
    results.update(data_settings)
    
    # Ini data
    data = load_data(**data_settings)
    
    # TRAIN MODELS
    # MLP
    name = "MLP"
    # We need to add the input size for the MLP
    HYPERPARAMS.update({"input_size": [data.x_train.shape[1]]})
    # Train and tune
    model, best_paras = train_val_tuner(
        data=data,
        model=MLP,
        parameters=HYPERPARAMS,
        name=name,
        num_combinations=RANDOM_SEARCH_N,
    )
    
    results.update(
        {
            "MISE " + name: mean_integrated_prediction_error(
                x = data.x_test,
                response = data.ground_truth,
                model = model,
            ),
        }
    )
    
    # FINISH
    
    # Add results
    add_dict(RES_FILE, results)