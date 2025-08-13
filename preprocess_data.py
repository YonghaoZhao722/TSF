import pandas as pd
import numpy as np
import os
from model.methods import load_rsf_data
from global_names import *

def preprocess_and_save_data():
    """
    Preprocess data for both SEER and WCH datasets and save them to avoid repeated processing
    """
    datasets = [SEER, WCH]
    
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset...")
        try:
            X, y = load_rsf_data(dataset_name)
            
            # Create preprocessed data directory if it doesn't exist
            os.makedirs('preprocessed_data', exist_ok=True)
            
            # Save preprocessed data
            X.to_csv(f'preprocessed_data/{dataset_name}_X.csv', index=False)
            # Convert y to DataFrame for CSV saving
            # Handle structured array format from load_rsf_data
            if y.dtype.names:  # It's a structured array
                y_df = pd.DataFrame({
                    'status': y['status'],
                    'survival_time': y['Survival.months']
                })
            else:  # It's a regular array
                y_df = pd.DataFrame(y, columns=['status', 'survival_time'] if y.shape[1] == 2 else [f'col_{i}' for i in range(y.shape[1])])
            y_df.to_csv(f'preprocessed_data/{dataset_name}_y.csv', index=False)
                
            print(f"Successfully preprocessed and saved {dataset_name} dataset")
            print(f"  - Features shape: {X.shape}")
            print(f"  - Targets shape: {y.shape}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

def load_preprocessed_data(dataset_name):
    """
    Load preprocessed data for a given dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (SEER or WCH)
        
    Returns:
    --------
    X : pandas.DataFrame
        Features
    y : numpy.ndarray
        Targets with status and survival time
    """
    try:
        X = pd.read_csv(f'preprocessed_data/{dataset_name}_X.csv')
        y_df = pd.read_csv(f'preprocessed_data/{dataset_name}_y.csv')
        y = y_df.values  # Convert back to numpy array
        return X, y
    except FileNotFoundError:
        print(f"Preprocessed data for {dataset_name} not found. Please run preprocess_and_save_data() first.")
        raise

if __name__ == "__main__":
    print("Starting data preprocessing...")
    preprocess_and_save_data()
    print("Data preprocessing completed!")