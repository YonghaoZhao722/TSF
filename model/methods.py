import pickle
import numpy as np
import pandas as pd
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


def sample_data(x, y, n, k=None, verbose=True):
    np.random.seed(1234)
    censor_rate = np.sum(y[1] == False) / len(y[1])
    censored_indices = np.where(y[1] == False)[0]
    uncensored_indices = np.where(y[1] == True)[0]

    num_censored = int(n * censor_rate)
    num_uncensored = n - num_censored

    censored_sample_indices = np.random.choice(censored_indices, num_censored, replace=False)
    uncensored_sample_indices = np.random.choice(uncensored_indices, num_uncensored, replace=False)

    sample_indices = np.concatenate([censored_sample_indices, uncensored_sample_indices])
    np.random.shuffle(sample_indices)  # shuffle the indices to mix censored and uncensored samples

    # If k is not None and num_uncensored is less than k, replace some censored samples with uncensored samples
    if k is not None and num_uncensored < k:
        additional_uncensored_indices = np.random.choice(uncensored_indices, k - num_uncensored, replace=False)
        censored_to_remove = np.random.choice(censored_sample_indices, k - num_uncensored, replace=False)
        sample_indices = np.setdiff1d(sample_indices, censored_to_remove, assume_unique=True)
        sample_indices = np.concatenate([sample_indices, additional_uncensored_indices])
        np.random.shuffle(sample_indices)  # shuffle the indices again after replacement
        num_uncensored = k  # update num_uncensored to k

    if verbose:
        print(f"uncensored:{num_uncensored}")

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
    
    # Standard Concordance index (C-index) using manual calculation
    try:
        # Use risk scores from last time point (or median time)
        if survival_probs.shape[0] > 0:
            # Use survival probability at median time as risk score (inverted)
            median_time_idx = len(survival_probs) // 2
            risk_scores = -survival_probs.iloc[median_time_idx].values  # Higher risk = lower survival
        else:
            risk_scores = np.zeros(len(y_test_time))
        
        c_index = _calculate_harrell_c_index(y_test_time, y_test_event, risk_scores)
        metrics['c_index'] = float(c_index) if np.isfinite(c_index) else np.nan
    except Exception as e:
        print(f"Warning: C-index calculation failed: {e}")
        metrics['c_index'] = 0.5
    
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