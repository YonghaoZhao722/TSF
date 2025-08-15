import pickle
import numpy as np
import pandas as pd
import os
import random
from global_names import *
from pycox.evaluation import EvalSurv


def one_hot_encoder(df, type=None):
    """
    convert binary 'cea' to one hot
    !!! FOR CONTINUOUS CEA ONLY!!!
    """
    df['cea_code'] = df['cea'].apply(lambda x: 1 if x > 2.5 else 0)
    df['cea_positive'] = (df['cea_code'] == 1).astype(int)
    df['cea_negative'] = (df['cea_code'] == 0).astype(int)
    df['cea_unknown'] = 0
    if type == 'T':
        df['cea_code'] = df['cea'].apply(lambda x: 1 if x > 2.5 else 0)
        df['T1'] = (df['T'] == 'T1').astype(int)
        df['T2'] = (df['T'] == 'T2').astype(int)
        df['T3'] = (df['T'] == 'T3').astype(int)
        df['T4'] = (df['T'] == 'T4').astype(int)
    return df

def load_rsf_data(dataset_name):
    rename_columns = {'gendercat': 'gender_code', 'gradecat': 'grade_code', 'tstage': 'T', 'lymcat': 'lymphcat',
                     'pni': 'PNI', 'OS': 'Survival.months', 'OSS': 'status'}

    if dataset_name == SEER:
        df = pd.read_csv("data/SEER.csv")
        X = df[['age', 'tumor.size', 'T', 'gender_code', 'grade_code',
                'PNI',
                'lymphcat',
                'cea_positive'
                ]]
        y = df[['status', 'Survival.months']]
        y = y.astype({'status': 'bool', 'Survival.months': 'float64'})
        y['status'] = y['status'].apply(lambda x: False if x == 0 else True)
        y = np.array(list(zip(y['status'], y['Survival.months'])),
                     dtype=[('status', np.bool_), ('Survival.months', np.float64)])
        X = X[['gender_code', 'age', 'tumor.size', 'grade_code', 'T', 'lymphcat', 'PNI', 'cea_positive']]
        print(f"data size: {len(X)}")
        return X, y

    else:
        df = pd.read_csv("data/stagel/train_hx.csv")
        # df = pd.read_csv("data/example.csv")  # please prepare your dataset here
        
        df = df.rename(columns=rename_columns)
        df['grade_code'] = df['grade_code'].apply(lambda x: 1 if x == 0 else 0)
        df = one_hot_encoder(df)
        X = df.drop(
            ['Survival.months', 'status', 'gender', 'tumor.site', 'grade', 'nolym', 'lvi', 'ID', 'logcea', 'cea_code',
             'cea', 'DFSS', 'DFS', 'CRCD',], axis=1)
        y = df[['status', 'Survival.months']]
        y = y.astype({'status': 'bool', 'Survival.months': 'float64'})
        y['status'] = y['status'].apply(lambda x: False if x == 0 else True)
        y = np.array(list(zip(y['status'], y['Survival.months'])),
                     dtype=[('status', np.bool_), ('Survival.months', np.float64)])
        X = X[['gender_code', 'age', 'tumor.size', 'grade_code', 'T', 'lymphcat', 'PNI', 'cea_positive']]
        print(f"data size: {len(X)}")
        return X, y


def storeForest(inputForest, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputForest, fw)
    fw.close()

def grapForest(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


def sample_data(x, y, n, k=10, verbose=True):
    np.random.seed(1234)
    
    # Handle structured array format
    if hasattr(y, 'dtype') and y.dtype.names:
        # y is a structured array, access via field names
        status_field = 'status'
        # Find the survival time field (could be 'Survival.months' or other)
        time_fields = [name for name in y.dtype.names if name != 'status']
        time_field = time_fields[0] if time_fields else 'Survival.months'
        
        status_data = y[status_field]
        censor_rate = np.sum(status_data == False) / len(status_data)
        censored_indices = np.where(status_data == False)[0]
        uncensored_indices = np.where(status_data == True)[0]
    else:
        # y is a tuple format (y[0] = status, y[1] = survival time)
        censor_rate = np.sum(y[1] == False) / len(y[1])
        censored_indices = np.where(y[1] == False)[0]
        uncensored_indices = np.where(y[1] == True)[0]

    # Calculate initial distribution based on original censor rate
    num_censored = int(n * censor_rate)
    num_uncensored = n - num_censored
    
    # Ensure minimum uncensored samples if k is specified
    if k is not None:
        # Calculate the maximum possible uncensored samples
        max_possible_uncensored = min(len(uncensored_indices), n)
        # Set target uncensored to minimum of k and what's possible
        target_uncensored = min(k, max_possible_uncensored)
        
        # If we need more uncensored samples than initially calculated
        if target_uncensored > num_uncensored:
            num_uncensored = target_uncensored
            num_censored = n - num_uncensored
            
            # Ensure we don't exceed available censored samples
            if num_censored > len(censored_indices):
                num_censored = len(censored_indices)
                num_uncensored = n - num_censored
    
    # Ensure we don't exceed available samples
    num_censored = min(num_censored, len(censored_indices))
    num_uncensored = min(num_uncensored, len(uncensored_indices))
    
    # Adjust if total exceeds n
    total_samples = num_censored + num_uncensored
    if total_samples > n:
        # Reduce proportionally
        ratio = n / total_samples
        num_censored = int(num_censored * ratio)
        num_uncensored = n - num_censored

    # Sample indices
    if num_censored > 0:
        censored_sample_indices = np.random.choice(censored_indices, num_censored, replace=False)
    else:
        censored_sample_indices = np.array([])
    
    if num_uncensored > 0:
        uncensored_sample_indices = np.random.choice(uncensored_indices, num_uncensored, replace=False)
    else:
        uncensored_sample_indices = np.array([])

    sample_indices = np.concatenate([censored_sample_indices, uncensored_sample_indices])
    np.random.shuffle(sample_indices)  # shuffle the indices to mix censored and uncensored samples

    if verbose:
        print(f"uncensored:{num_uncensored}, censored:{num_censored}, total:{len(sample_indices)}")

    # Return sampled data in the same format
    if hasattr(y, 'dtype') and y.dtype.names:
        # Return structured array format
        return x.iloc[sample_indices], y[sample_indices]
    else:
        # Return tuple format
        return x[sample_indices], (y[0][sample_indices], y[1][sample_indices])


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
        # Convert back to structured array format to maintain compatibility
        y = np.array(list(zip(y_df['status'], y_df['survival_time'])),
                     dtype=[('status', np.bool_), ('Survival.months', np.float64)])
        return X, y
    except FileNotFoundError:
        print(f"Preprocessed data for {dataset_name} not found. Please run preprocess_data.py first.")
        raise


def _calculate_harrell_c_index(time, event, risk_scores):
    """
    Calculate Harrell's C-index manually
    """
    n = len(time)
    concordant = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Only consider pairs where one has an event and earlier time
            if event[i] and time[i] < time[j]:
                # Risk should be higher for patient i (earlier event)
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                total_pairs += 1
            elif event[j] and time[j] < time[i]:
                # Risk should be higher for patient j (earlier event)
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                total_pairs += 1
    
    return concordant / total_pairs if total_pairs > 0 else 0.5


def calculate_comprehensive_metrics(model, X_test, y_test):
    """
    Calculate comprehensive survival evaluation metrics
    
    Parameters:
    -----------
    model : Forest
        Trained survival model
    X_test : pandas.DataFrame
        Test features
    y_test : numpy.ndarray
        Test targets with status and survival time
        
    Returns:
    --------
    dict : Dictionary containing evaluation metrics
    """
    # Make predictions
    model.predict(X_test)
    
    # Extract event indicator and time from structured array
    y_test_event = np.array([entry[0] for entry in y_test], dtype=bool)
    y_test_time = np.array([entry[1] for entry in y_test])
    
    # Convert cumulative hazard to survival probabilities
    survival_probs = np.exp(-model.surv)
    
    # Ensure proper DataFrame format for pycox
    if not isinstance(survival_probs, pd.DataFrame):
        survival_probs = pd.DataFrame(survival_probs)
    
    # Get time points from model (usually from training data)
    if hasattr(model, 'unique_times') and model.unique_times:
        time_points = sorted(list(model.unique_times))
    else:
        # Use quantiles of test times as approximation
        time_points = np.percentile(y_test_time, np.linspace(10, 90, 10))
    
    # Ensure time_points is sorted and convert to list
    time_points = sorted([float(t) for t in time_points if t > 0])
    
    # Set proper index for survival_probs (should be time points)
    if len(time_points) == survival_probs.shape[0]:
        survival_probs.index = time_points
    else:
        # If dimensions don't match, create appropriate time index
        survival_probs.index = np.linspace(
            min(time_points), max(time_points), survival_probs.shape[0]
        )
    
    # Calculate metrics
    metrics = {}
    
    # Time-dependent Concordance index (CTD) using Antolini method
    try:
        ev_ctd = EvalSurv(survival_probs, y_test_time, y_test_event)
        ctd_value = ev_ctd.concordance_td('antolini')
        metrics['ctd'] = float(ctd_value) if np.isfinite(ctd_value) else np.nan
    except Exception as e:
        print(f"Warning: CTD calculation failed: {e}")
        metrics['ctd'] = np.nan
    
    # Time-dependent AUC
    try:
        # Calculate time-dependent AUC at multiple time points
        observed_times = y_test_time[y_test_event]  # Only event times
        if len(observed_times) > 3:
            # Use quartiles of observed event times
            eval_times = np.percentile(observed_times, [25, 50, 75])
        else:
            # Use quantiles of all times if not enough events
            eval_times = np.percentile(y_test_time, [25, 50, 75])
        
        # Ensure times are within valid range and positive
        max_time = np.max(y_test_time)
        eval_times = eval_times[eval_times < max_time * 0.9]
        eval_times = eval_times[eval_times > 0]
        
        td_aucs = []
        for t in eval_times:
            # Define binary outcomes at time t
            # Positive: event occurred by time t
            # Negative: event-free at time t (censored after t or alive beyond t)
            outcomes = []
            scores = []
            
            for i in range(len(y_test_time)):
                if y_test_event[i] and y_test_time[i] <= t:
                    # Event occurred by time t
                    outcomes.append(1)
                elif y_test_time[i] > t:
                    # Survived beyond time t (either censored or alive)
                    outcomes.append(0)
                # Skip cases where censored before time t (uninformative)
                else:
                    continue
                
                # Get survival probability at time t as score
                if len(survival_probs.index) > 0:
                    closest_idx = np.argmin(np.abs(survival_probs.index - t))
                    # Use 1 - survival probability as risk score (higher = more likely to have event)
                    risk_score = 1 - survival_probs.iloc[closest_idx, i]
                    scores.append(risk_score)
                else:
                    scores.append(0.5)
            
            # Calculate AUC if we have both classes
            if len(set(outcomes)) > 1 and len(outcomes) > 5:
                try:
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(outcomes, scores)
                    td_aucs.append(auc)
                except ImportError:
                    # Manual AUC calculation if sklearn not available
                    pos_scores = [scores[i] for i in range(len(scores)) if outcomes[i] == 1]
                    neg_scores = [scores[i] for i in range(len(scores)) if outcomes[i] == 0]
                    if len(pos_scores) > 0 and len(neg_scores) > 0:
                        # Count concordant pairs
                        concordant = sum(1 for p in pos_scores for n in neg_scores if p > n)
                        total_pairs = len(pos_scores) * len(neg_scores)
                        auc = concordant / total_pairs if total_pairs > 0 else 0.5
                        td_aucs.append(auc)
                except Exception:
                    continue
        
        # Average AUC across time points
        if td_aucs:
            avg_td_auc = np.mean(td_aucs)
            metrics['time_dependent_auc'] = float(avg_td_auc) if np.isfinite(avg_td_auc) else np.nan
        else:
            metrics['time_dependent_auc'] = np.nan
            
    except Exception as e:
        print(f"Warning: Time-dependent AUC calculation failed: {e}")
        metrics['time_dependent_auc'] = np.nan
    
    # Integrated Brier Score
    try:
        # Create structured arrays for pycox
        y_train_struct = np.array([(True, 1.0)], dtype=[('event', bool), ('time', float)])  # Dummy
        y_test_struct = np.array(
            [(bool(event), float(time)) for event, time in zip(y_test_event, y_test_time)],
            dtype=[('event', bool), ('time', float)]
        )
        
        # Get evaluation time points - use quantiles of observed times
        observed_times = y_test_time[y_test_event]  # Only event times
        if len(observed_times) > 3:
            eval_times = np.percentile(observed_times, [20, 40, 60, 80])
        else:
            eval_times = np.percentile(y_test_time, [25, 50, 75])
        
        # Ensure times are within valid range
        max_time = np.max(y_test_time)
        eval_times = eval_times[eval_times < max_time * 0.9]
        eval_times = eval_times[eval_times > 0]
        
        if len(eval_times) >= 2:
            # Prepare survival probabilities for the evaluation times
            survival_at_times = []
            for t in eval_times:
                # Find closest time point in our survival function
                if len(survival_probs.index) > 0:
                    closest_idx = np.argmin(np.abs(survival_probs.index - t))
                    survival_at_times.append(survival_probs.iloc[closest_idx].values)
                else:
                    survival_at_times.append(np.ones(len(y_test_time)) * 0.5)
            
            survival_matrix = np.array(survival_at_times).T  # Shape: (n_samples, n_times)
            
            # Use scikit-survival's integrated_brier_score if available
            try:
                from sksurv.metrics import integrated_brier_score
                from sksurv.util import Surv
                
                # Convert to sksurv format
                y_sksurv = Surv.from_arrays(y_test_event, y_test_time)
                ibs = integrated_brier_score(y_sksurv, y_sksurv, survival_matrix, eval_times)
                metrics['integrated_brier_score'] = float(ibs) if np.isfinite(ibs) else np.nan
            except ImportError:
                # Fallback: use pycox but with proper setup
                try:
                    ev_ibs = EvalSurv(survival_probs, y_test_time, y_test_event)
                    ev_ibs.add_censor_est(censor_surv='km', censor_durations=y_test_time, censor_events=y_test_event)
                    ibs = ev_ibs.integrated_brier_score(eval_times)
                    metrics['integrated_brier_score'] = float(ibs) if np.isfinite(ibs) else np.nan
                except:
                    metrics['integrated_brier_score'] = np.nan
        else:
            metrics['integrated_brier_score'] = np.nan
            
    except Exception as e:
        print(f"Warning: Integrated Brier score calculation failed: {e}")
        metrics['integrated_brier_score'] = np.nan
    
    return metrics


def set_all_seeds(seed=1234):
    """
    Set seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed (int): The random seed to use (default: 1234)
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set PyTorch seeds if PyTorch is available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make PyTorch deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"PyTorch seeds set to {seed}")
    except ImportError:
        # PyTorch not available, skip
        pass
    
    # Set joblib to use single thread for deterministic results
    # This can be overridden if needed, but single-threaded is most reproducible
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    print(f"All random seeds set to {seed}")


def set_joblib_deterministic():
    """
    Configure joblib for deterministic parallel processing.
    This trades some performance for reproducibility.
    """
    # Force single-threaded execution for maximum reproducibility
    os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/temp/joblib'
    return {'n_jobs': 1}  # Return joblib kwargs for deterministic execution