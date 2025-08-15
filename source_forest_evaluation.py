import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model.methods import load_preprocessed_data, grapForest, calculate_comprehensive_metrics, set_all_seeds, sample_data
from global_names import *
import os
from datetime import datetime

def run_source_forest_evaluation():
    """
    Evaluate source forest performance across different sample sizes
    """
    # Set all random seeds for reproducibility
    set_all_seeds(1234)
    
    dataset_name = WCH
    
    # Load preprocessed data
    try:
        X_full, y_full = load_preprocessed_data(dataset_name)
        print(f"Loaded preprocessed {dataset_name} data successfully")
        print(f"Original dataset size: {len(X_full)}")
    except FileNotFoundError:
        print("Preprocessed data not found. Please run preprocess_data.py first.")
        exit(1)
    
    # Sample sizes to evaluate
    sample_sizes = ['unlimited', 500, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20]
    
    # Initialize result storage
    aggregated_results = []
    detailed_results = []
    
    print(f"Starting source forest evaluation with {len(sample_sizes)} sample sizes")
    print(f"Sample sizes: {sample_sizes}")
    
    # Load the source forest model
    try:
        source_forest = grapForest("rsf_models/source_forest.pkl")
        print("Source forest loaded successfully")
    except FileNotFoundError:
        print("Source forest model not found at rsf_models/source_forest.pkl")
        exit(1)
    
    # Evaluation loop
    for i, sample_size in enumerate(sample_sizes):
        print(f"\n{'='*60}")
        print(f"Evaluation {i+1}/{len(sample_sizes)}")
        print(f"Sample size: {sample_size}")
        print(f"{'='*60}")
        
        # Prepare data for this sample size
        if sample_size == 'unlimited':
            X, y = X_full, y_full
            actual_sample_size = len(X)
            print(f"Using full dataset: {actual_sample_size} samples")
        else:
            if sample_size > len(X_full):
                print(f"Warning: Requested sample size {sample_size} is larger than dataset size {len(X_full)}")
                X, y = X_full, y_full
                actual_sample_size = len(X)
            else:
                X, y = sample_data(X_full, y_full, sample_size, k=10)
                actual_sample_size = len(X)
                print(f"Sampled {actual_sample_size} samples from full dataset")
        
        # Cross-validation setup - adjust folds based on sample size
        if actual_sample_size < 100:
            n_splits = 5
        else:
            n_splits = 10
        
        print(f"Using {n_splits}-fold cross-validation for {actual_sample_size} samples")
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state= 1234)
        
        # Initialize metrics storage for this sample size
        fold_metrics = {
            'ctd': [],
            'time_dependent_auc': [],
            'integrated_brier_score': []
        }
        
        # Cross-validation loop
        for fold, (train_index, test_index) in enumerate(kf.split(X, y['status'])):
            print(f"  Processing fold {fold + 1}/{n_splits}...")
            
            # Split data for this fold (we only use test data for evaluation)
            X_test = X.iloc[test_index]
            y_test = y[test_index]
            
            # Calculate metrics using source forest (no training/fine-tuning)
            print(f"    Calculating source forest metrics...")
            try:
                metrics = calculate_comprehensive_metrics(source_forest, X_test, y_test)
                
                # Store fold results
                fold_metrics['ctd'].append(metrics['ctd'])
                fold_metrics['time_dependent_auc'].append(metrics['time_dependent_auc'])
                fold_metrics['integrated_brier_score'].append(metrics['integrated_brier_score'])
                
                # Store detailed results
                detailed_results.append({
                    'sample_size': sample_size,
                    'actual_sample_size': actual_sample_size,
                    'n_splits': n_splits,
                    'fold': fold + 1,
                    'ctd': metrics['ctd'],
                    'time_dependent_auc': metrics['time_dependent_auc'],
                    'integrated_brier_score': metrics['integrated_brier_score']
                })
                
                # Print current fold results with safe formatting
                def safe_format(value, name):
                    try:
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            return f"{value:.4f}"
                        else:
                            return "N/A"
                    except:
                        return "N/A"
                        
                print(f"    Source forest metrics:")
                print(f"      CTD: {safe_format(metrics['ctd'], 'CTD')}")
                print(f"      TD-AUC: {safe_format(metrics['time_dependent_auc'], 'TD-AUC')}")
                print(f"      IBS: {safe_format(metrics['integrated_brier_score'], 'IBS')}")
                
            except Exception as e:
                print(f"    Error calculating metrics for fold {fold + 1}: {str(e)}")
                # Store NaN values for failed fold
                fold_metrics['ctd'].append(np.nan)
                fold_metrics['time_dependent_auc'].append(np.nan)
                fold_metrics['integrated_brier_score'].append(np.nan)
                
                detailed_results.append({
                    'sample_size': sample_size,
                    'actual_sample_size': actual_sample_size,
                    'n_splits': n_splits,
                    'fold': fold + 1,
                    'ctd': np.nan,
                    'time_dependent_auc': np.nan,
                    'integrated_brier_score': np.nan
                })
        
        # Calculate aggregated results for this sample size
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
        
        # Calculate means and stds for metrics
        ctd_mean, ctd_std = safe_mean_std(fold_metrics['ctd'])
        td_auc_mean, td_auc_std = safe_mean_std(fold_metrics['time_dependent_auc'])
        ibs_mean, ibs_std = safe_mean_std(fold_metrics['integrated_brier_score'])
        
        # Store aggregated results
        aggregated_results.append({
            'sample_size': sample_size,
            'actual_sample_size': actual_sample_size,
            'n_splits': n_splits,
            'ctd_mean': ctd_mean,
            'ctd_std': ctd_std,
            'td_auc_mean': td_auc_mean,
            'td_auc_std': td_auc_std,
            'ibs_mean': ibs_mean,
            'ibs_std': ibs_std
        })
        
        # Print summary for this sample size
        print(f"\n  Summary for sample_size={sample_size}:")
        def safe_format_result(mean_val, std_val):
            try:
                if np.isfinite(mean_val) and np.isfinite(std_val):
                    return f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    return "N/A"
            except:
                return "N/A"
        
        print(f"    Source forest metrics:")
        print(f"      CTD: {safe_format_result(ctd_mean, ctd_std)}")
        print(f"      TD-AUC: {safe_format_result(td_auc_mean, td_auc_std)}")
        print(f"      IBS: {safe_format_result(ibs_mean, ibs_std)}")
    
    return aggregated_results, detailed_results

def save_results(aggregated_results, detailed_results):
    """
    Save results to CSV files
    """
    # Create results directory if it doesn't exist
    os.makedirs('source_evaluation_results', exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save aggregated results
    agg_df = pd.DataFrame(aggregated_results)
    agg_filename = f'source_evaluation_results/source_aggregated_results_{timestamp}.csv'
    agg_df.to_csv(agg_filename, index=False)
    print(f"\nAggregated results saved to: {agg_filename}")
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_filename = f'source_evaluation_results/source_detailed_results_{timestamp}.csv'
    detailed_df.to_csv(detailed_filename, index=False)
    print(f"Detailed results saved to: {detailed_filename}")
    
    return agg_filename, detailed_filename

def analyze_results(aggregated_results):
    """
    Analyze and display results
    """
    agg_df = pd.DataFrame(aggregated_results)
    
    # Filter out rows with NaN values for analysis
    valid_results = agg_df.dropna(subset=['td_auc_mean'])
    
    if len(valid_results) == 0:
        print("No valid results to analyze.")
        return
    
    # Sort by Time-dependent AUC (higher is better)
    sorted_results = valid_results.sort_values('td_auc_mean', ascending=False)
    
    print(f"\nSource Forest Performance Analysis:")
    print("=" * 80)
    
    # Display all results sorted by performance
    print(f"Results ranked by Time-dependent AUC (higher is better):")
    print("-" * 80)
    
    for i, (_, row) in enumerate(sorted_results.iterrows()):
        print(f"{i+1}. Sample size: {row['sample_size']} (actual: {row['actual_sample_size']})")
        
        # Helper function for safe formatting
        def safe_format_metric(mean_val, std_val):
            try:
                if pd.notna(mean_val) and pd.notna(std_val):
                    return f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    return "N/A"
            except:
                return "N/A"
        
        print(f"   CTD: {safe_format_metric(row['ctd_mean'], row['ctd_std'])}")
        print(f"   TD-AUC: {safe_format_metric(row['td_auc_mean'], row['td_auc_std'])}")
        print(f"   IBS: {safe_format_metric(row['ibs_mean'], row['ibs_std'])}")
        print()
    
    # Find best performing sample size
    best_result = sorted_results.iloc[0]
    print(f"Best performing configuration:")
    print(f"  Sample size: {best_result['sample_size']} (actual: {best_result['actual_sample_size']})")
    print(f"  TD-AUC: {best_result['td_auc_mean']:.4f} ± {best_result['td_auc_std']:.4f}")
    print(f"  CTD: {best_result['ctd_mean']:.4f} ± {best_result['ctd_std']:.4f}")
    print(f"  IBS: {best_result['ibs_mean']:.4f} ± {best_result['ibs_std']:.4f}")

if __name__ == "__main__":
    print("Starting Source Forest Evaluation")
    print("=" * 60)
    
    # Run evaluation
    aggregated_results, detailed_results = run_source_forest_evaluation()
    
    # Save results
    agg_file, detailed_file = save_results(aggregated_results, detailed_results)
    
    # Analyze results
    analyze_results(aggregated_results)
    
    print("\n" + "=" * 60)
    print("SOURCE FOREST EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Total sample sizes tested: {len(aggregated_results)}")
    print(f"Total folds processed: {len(detailed_results)}")
    print(f"\nResults saved:")
    print(f"  - Aggregated: {agg_file}")
    print(f"  - Detailed: {detailed_file}")