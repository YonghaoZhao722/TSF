import pickle
import numpy as np
import pandas as pd
from global_names import *


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
        # df = pd.read_csv("data/example.csv")  # please upload your dataset here

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