import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_csv_to_numpy(file_path):
    df = pd.read_csv(file_path, header=None, index_col=None, sep="\t")
    return np.array(df.values)

def calculate_mae_mse(base_path, analysis_path, file_name, pbar):
    os.makedirs(analysis_path, exist_ok=True)
        
    analytic_path = os.path.join(base_path, 'analytic.csv')
    analytic_solution = load_csv_to_numpy(analytic_path)
        
    for j, model in enumerate(MODELS):
        model_path = os.path.join(base_path, model)

        for i in INIT_SEEDS:
            
            file_path = os.path.join(beta_path, f"seed_{i}", file_name)
            data = load_csv_to_numpy(file_path)

        pbar.update(1)

    pd.DataFrame(np.array((MAE_mean, MAE_std, MSE_mean, MSE_std)).T, columns=['MAE mean', 'MAE std', 'MSE mean', 'MSE std']).to_csv(os.path.join(analysis_path, "1_or_10_maemse50.csv"), index=False)
            

NUM_SEEDS = 100
INIT_SEEDS = np.array(range(NUM_SEEDS))
MODELS = ["FLS", "FLW", "PINN"]

TOTAL_COMBINATIONS = len(MODELS)*NUM_SEEDS
base_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    progress_bar = tqdm(total=TOTAL_COMBINATIONS, ncols=100)

    experiment_name = "THREE_MODELS"
    
    data_path = os.path.join(base_dir, 'results', experiment_name, '32bit')
    analysis_path = os.path.join(base_dir, 'analysis', experiment_name)

    calculate_mae_mse(data_path, analysis_path, "error.csv", progress_bar)