import csv
from collections import defaultdict
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from joblib import Parallel, delayed
from model.TransferTree import Tree
import os
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/temp/joblib'  # This is compatibility settings for parallel computing



def bootstrap_data(X, y, tree_random):
    n_samples = X.shape[0]
    indices = tree_random.choice(n_samples, n_samples, replace=True)
    indices = np.unique(indices)
    X_bootstrapped = X.iloc[indices]
    y_bootstrapped = y[indices]
    while len(np.unique(y_bootstrapped['status'])) < 2:
        indices = tree_random.choice(n_samples, n_samples, replace=True)
        indices = np.unique(indices)
        X_bootstrapped = X.iloc[indices]
        y_bootstrapped = y[indices]
    return X_bootstrapped, y_bootstrapped


def get_probabilities(features_order):
    data = defaultdict(dict)
    with open('dp.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            depth = int(row[0])
            feature = row[1]
            value = float(row[2])
            data[depth][feature] = value

    for depth, features in data.items():
        for feature in features_order:
            if feature not in features:
                min_value = min(features.values())
                features[feature] = min_value / len(features)

        features = {feature: features[feature] for feature in features_order}

        sum_values = sum(features.values())

        for feature in features_order:
            features[feature] /= sum_values

        data[depth] = list(features.values())
    return data


def get_step_function(Y_pred):
    Y_pred_df = pd.DataFrame(Y_pred).transpose()
    Y_pred_df = Y_pred_df.sort_index()
    Y_pred_df = Y_pred_df.ffill()
    return Y_pred_df


class Forest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=6,
                 min_samples_leaf=0, max_features=None, random_state=1234,
                 max_transfer_depth=None,):
        self.unique_times = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = np.random.RandomState(random_state)
        self.estimators = None
        self.bootstrap = True
        self.max_transfer_depth = max_transfer_depth
        self.probabilities = None

    def fit_one_tree(self, args):

        X, y, (idx, tree_random_state) = args
        tree_random = np.random.RandomState(tree_random_state)
        X_bootstrapped, y_bootstrapped = bootstrap_data(X, y, tree_random)
        estimator = Tree(max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         min_samples_leaf=self.min_samples_leaf,
                         max_feature=self.max_features,
                         random_state=tree_random_state,
                         max_transfer_depth=self.max_transfer_depth,
                         forest=True,
                         )
        if self.probabilities is not None:
            estimator.probabilities = self.probabilities
        estimator.fit(X_bootstrapped, y_bootstrapped)
        return estimator

    def fit(self, X, y):
        if self.estimators:
            # Randomly sample with replacement n_estimators number of estimators from self.estimators
            # selected_estimators = self.random_state.choice(self.estimators, self.n_estimators, replace=True)
            selected_estimators = []

            while len(selected_estimators) < self.n_estimators:
                estimator = self.random_state.choice(self.estimators)

                # Skip trees with only one node
                if len(estimator.nodes) == 1:
                    continue

                selected_estimators.append(estimator)

            selected_estimators = np.array(selected_estimators)
            for estimators in selected_estimators:
                estimators.min_samples_split=self.min_samples_split
                estimators.min_samples_leaf=self.min_samples_leaf
                estimators.max_feature=self.max_features
                estimators.max_transfer_depth=self.max_transfer_depth

            self.estimators = []
            # Train using the selected estimators
            self.estimators = (
                Parallel(n_jobs=-1)(
                    delayed(estimator.fit)(X, y) for estimator in selected_estimators
                ))
        else:
            self.estimators = []
            tree_random_states = [(idx, self.random_state.randint(0, 10000)) for idx in range(self.n_estimators)]
            self.estimators = (
                Parallel(n_jobs=-1)(
                    delayed(self.fit_one_tree)((X, y, rs)) for rs in tree_random_states
                ))

    def predict(self, X):
        predictions = None
        all_unique_value = set()
        for estimator in self.estimators:
            # Some trees only have one node after fitting, these trees won't affect predictions, remove them to speed up prediction
            if len(estimator.nodes) == 1:
                continue
            Y_pred, unique_times = estimator.predict(X)
            all_unique_value.update(unique_times)
            Y_pred_df = get_step_function(Y_pred)

            if predictions is None:
                predictions = Y_pred_df.copy()  # During first iteration, initialize predictions as a copy of the first prediction DataFrame
            else:
                # Use reindex and ffill to align indices of the two DataFrames and perform forward fill
                predictions = predictions.reindex(index=predictions.index.union(Y_pred_df.index)).ffill().add(
                    Y_pred_df.reindex(index=predictions.index.union(Y_pred_df.index)).ffill())

        predictions /= len(self.estimators)
        self.unique_times = all_unique_value
        self.surv = predictions

    def ctd(self, x, y):
        self.predict(x)
        survival_df = np.exp(-self.surv)
        y_test_event = np.array([entry[0] for entry in y], dtype=bool)
        y_test_time = np.array([entry[1] for entry in y])
        ev = EvalSurv(survival_df, y_test_time, y_test_event)
        current_ctd = ev.concordance_td('antolini')
        return current_ctd


    def load_softmax_probabilities(self,features_order):
        self.probabilities = get_probabilities(features_order)