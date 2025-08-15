import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model.methods import load_preprocessed_data, grapForest, calculate_comprehensive_metrics, set_all_seeds
from global_names import *

if __name__ == "__main__":
    # Set all random seeds for reproducibility
    set_all_seeds(1234)
    
    dataset_name = WCH
    
    # Load preprocessed data
    try:
        X, y = load_preprocessed_data(dataset_name)
        print(f"Loaded preprocessed {dataset_name} data successfully")
    except FileNotFoundError:
        print("Preprocessed data not found. Please run preprocess_data.py first.")
        exit(1)
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
    
    # Initialize metrics storage
    all_metrics = {
        'ctd': [],
        'time_dependent_auc': [],
        'integrated_brier_score': []
    }

    for fold, (train_index, test_index) in enumerate(kf.split(X, y['status'])):
        print(f"Processing fold {fold + 1}/10...")
        
        rsf = grapForest(f"rsf_models/source_forest.pkl")

        rsf.n_estimators = 20
        rsf.min_samples_split = 6
        rsf.min_samples_leaf = 2
        rsf.max_features = 2
        rsf.cc = 1
        rsf.deterministic = True

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rsf.fit(X_train, y_train)
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(rsf, X_test, y_test)
        
        # Store metrics
        for key in all_metrics.keys():
            all_metrics[key].append(metrics[key])
        
        # Print current fold results with safe formatting
        def safe_format(value, name):
            try:
                if isinstance(value, (int, float)) and np.isfinite(value):
                    return f"{value:.4f}"
                else:
                    return "N/A"
            except:
                return "N/A"
                
        print(f"  CTD: {safe_format(metrics['ctd'], 'CTD')}")
        print(f"  TD-AUC: {safe_format(metrics['time_dependent_auc'], 'TD-AUC')}")
        print(f"  Integrated Brier Score: {safe_format(metrics['integrated_brier_score'], 'IBS')}")

    # Print final results  
    print("\n" + "="*50)
    print("FINAL RESULTS (Mean ± Std)")
    print("="*50)
    
    def safe_mean_std(values):
        """Calculate mean and std, handling NaN values"""
        clean_values = []
        for v in values:
            try:
                if isinstance(v, (int, float)) and np.isfinite(v):
                    clean_values.append(float(v))
            except:
                continue
        if clean_values:
            return np.mean(clean_values), np.std(clean_values)
        else:
            return np.nan, np.nan
    
    def safe_format_result(mean_val, std_val):
        """Safely format mean ± std results"""
        try:
            if np.isfinite(mean_val) and np.isfinite(std_val):
                return f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                return "N/A"
        except:
            return "N/A"
    
    ctd_mean, ctd_std = safe_mean_std(all_metrics['ctd'])
    print(f"CTD (Time-dependent): {safe_format_result(ctd_mean, ctd_std)}")
    
    td_auc_mean, td_auc_std = safe_mean_std(all_metrics['time_dependent_auc'])
    print(f"TD-AUC (Time-dependent): {safe_format_result(td_auc_mean, td_auc_std)}")
    
    ibs_mean, ibs_std = safe_mean_std(all_metrics['integrated_brier_score'])
    print(f"Integrated Brier Score: {safe_format_result(ibs_mean, ibs_std)}")
