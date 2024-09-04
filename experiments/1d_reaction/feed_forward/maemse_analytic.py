import os
from pinntorch import *
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_csv_to_numpy(file_path):
    df = pd.read_csv(file_path, header=None, index_col=None, sep="\t")
    return np.array(df.values)

def calculate_mae_mse(base_path, analysis_path, file_name, pbar):
    os.makedirs(analysis_path, exist_ok=True)
    # for every slider position we store mean and std
    MAE_mean = np.zeros(len(BETA_VALUES))
    MAE_std = np.zeros(len(BETA_VALUES))
    MSE_mean = np.zeros(len(BETA_VALUES))
    MSE_std = np.zeros(len(BETA_VALUES))
        
    analytic_path = os.path.join(base_path, 'analytic.csv')
    analytic_solution = load_csv_to_numpy(analytic_path)
        
    for j, beta in enumerate(MODELS):
        beta_path = os.path.join(base_path, f"beta_{beta}")

        MAE_single_means = np.zeros(50)
        MSE_single_means = np.zeros(50)

        for i in INIT_SEEDS:
            
            file_path = os.path.join(beta_path, f"seed_{i}", file_name)
            predicted_solution = load_csv_to_numpy(file_path)

            diff = predicted_solution - analytic_solution
            i -= 20
            # MAE and MSE calculation
            MAE_single_means[i] = np.mean(np.abs(diff))
            MSE_single_means[i] = np.mean(np.square(diff))

        MAE_mean[j] = np.mean(MAE_single_means)
        MAE_std[j] = np.std(MAE_single_means)
        MSE_mean[j] = np.mean(MSE_single_means)
        MSE_std[j] = np.std(MSE_single_means)

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

    calculate_mae_mse(data_path, analysis_path, "test_prediction.csv", progress_bar)