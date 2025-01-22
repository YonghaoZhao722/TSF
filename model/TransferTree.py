from collections import defaultdict

import pandas as pd
from lifelines import NelsonAalenFitter
from pycox.evaluation import EvalSurv
from sksurv.compare import compare_survival
import numpy as np




def _compute_unique_times(timelines):
    unique_times_set = set()
    for timeline in timelines:
        unique_times_set.update(timeline)

    unique_times = list(unique_times_set)

    # Sorting the unique_times list (optional)
    unique_times.sort()
    return unique_times


def _calculate_logrank(y, group_indicator):
    try:
        chisq, pval = compare_survival(y, group_indicator, return_stats=False)
    # except LinAlgError:
    except ValueError:
        chisq = 0

    return chisq


class Node:
    _index_counter = 0

    def __init__(self,
                 feature=None,
                 threshold=None,
                 value=None,
                 left=None,
                 right=None,
                 timeline=None,
                 depth=None,
                 ):
        self.index = Node._index_counter
        Node._index_counter += 1  # Increment index counter for the next node
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.timeline = timeline
        self.leaf = False
        self.depth = depth

    def _predict(self, y):
        naf = NelsonAalenFitter()
        naf.fit(y['Survival.months'], event_observed=y['status'])
        self.value=naf.cumulative_hazard_
        self.timeline = naf.timeline

        self.leaf = True
        self.left=None
        self.right=None

def bootstrap_data(X, y, random_state):
    random_state = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    indices = random_state.choice(n_samples, n_samples, replace=True)
    indices = np.unique(indices)
    X_bootstrapped = X.iloc[indices]
    y_bootstrapped = y[indices]
    while len(np.unique(y_bootstrapped['status'])) < 2:
        indices = random_state.choice(n_samples, n_samples, replace=True)
        indices = np.unique(indices)
        X_bootstrapped = X.iloc[indices]
        y_bootstrapped = y[indices]
    return X_bootstrapped, y_bootstrapped


class Tree:
    def __init__(self, max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=0,
                 random_state=1234,
                 max_feature=None,
                 forest=False,
                 max_transfer_depth=None, ):
        self.n_features_ = None
        self.tree_ = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_feature = max_feature
        self.random_state = random_state
        self.forest = forest
        self.nodes = []
        np.random.seed(self.random_state)
        self.max_transfer_depth = max_transfer_depth
        self.probabilities = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        if self.tree_ is None:
            self.tree_ = self._grow_tree(X, y)
            Node._index_counter = 0
        else:
            X, y = bootstrap_data(X, y, self.random_state)
            self.retrain(X, y)
            Node._index_counter = 0

        return self

    def _grow_tree(self, X, y, depth=0, ):
        n_samples = len(y)
        node = Node(depth=depth)
        self.nodes.append(node)
        if depth == self.max_depth or n_samples < self.min_samples_split:
            node._predict(y)
            return node

        # Perform best split
        feature, threshold, leaf = self._best_split(X, y, node=node)
        if leaf:
            node._predict(y)
            return node
        if feature:
            if len(np.unique(X.iloc[:, feature])) == 2:  # Check if the feature is binary
                indices_left = X.iloc[:, feature] == 0
            else:
                indices_left = X.iloc[:, feature] <= threshold
            X_left, y_left = X.loc[indices_left], y[indices_left]
            X_right, y_right = X.loc[~indices_left], y[~indices_left]
            if len(y_left) >= self.min_samples_leaf and len(y_right) >= self.min_samples_leaf:
                node.feature = feature
                node.threshold = threshold
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
            else:
                node._predict(y)
        else:
            node._predict(y)
        return node

    def retrain(self, X, y):
        if self.tree_ is not None:
            self.tree_ = self._retrain_tree(self.tree_, X, y, depth=0)
        else:
            raise ValueError("The tree has not been trained yet. Please call fit() method first.")

    def _retrain_tree(self, node, X, y, depth):
        n_samples = len(y)
        if depth == self.max_depth or n_samples < self.min_samples_split:
            node._predict(y)
            return node

        if node.feature is not None:
            if self.max_transfer_depth is None or depth < self.max_transfer_depth:

                best_feature, best_threshold, leaf = self._best_split(X, y, node=node)

                if leaf:
                    node._predict(y)
                    node.left = None
                    node.right = None
                    return node
                else:
                    node.threshold = best_threshold

        # Recursively retrain left and right child nodes if not beyond max_transfer_depth
        if not node.leaf:
            if len(np.unique(X.iloc[:, node.feature])) == 2:  # Check if the feature is binary
                left_indices = X.iloc[:, node.feature] == 0
                right_indices = X.iloc[:, node.feature] == 1
            else:
                left_indices = X.iloc[:, node.feature] < node.threshold
                right_indices = X.iloc[:, node.feature] >= node.threshold

            X_left, y_left = X.loc[left_indices], y[left_indices]
            X_right, y_right = X.loc[right_indices], y[right_indices]
            if (np.unique(left_indices).shape[0] == 1) or (np.unique(right_indices).shape[0] == 1):
                node._predict(y)
                return node

                # all censored
            if len(np.unique(y_left['status'])) == 1 or len(np.unique(y_right['status'])) == 1:
                node._predict(y)
                return node

            if self.max_transfer_depth is None or depth < self.max_transfer_depth:
                node.left = self._retrain_tree(node.left, X_left, y_left, depth + 1)
            else:
                node.left = self._grow_tree(X_left, y_left, depth + 1)

            if self.max_transfer_depth is None or depth < self.max_transfer_depth:
                node.right = self._retrain_tree(node.right, X_right, y_right, depth + 1)
            else:
                node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node

    def splitter(self, X, y, selected_features):
        leaf = False
        best_logrank = 0
        best_feature, best_threshold = None, None
        for feature in selected_features:
            thresholds = np.unique(X.iloc[:, feature])
            if len(thresholds) == 2:  # If the feature is a binary variable
                for threshold in thresholds:  # Directly divide into two groups according to 0 and 1
                    left_indices = X.iloc[:, feature] == threshold
                    group_indicator = []
                    if len(y[left_indices]) >= self.min_samples_leaf and len(y[~left_indices]) >= self.min_samples_leaf:
                        for is_left in left_indices:
                            if is_left:
                                group_indicator.append(1)  # Sample of left child node
                            else:
                                group_indicator.append(0)  # Sample of right child node
                        logrank = _calculate_logrank(y, group_indicator)
                        if logrank > best_logrank:
                            best_logrank = logrank
                            best_feature = feature
                            best_threshold = threshold
                    elif len(y[left_indices]) < self.min_samples_leaf and len(y[~left_indices]) < self.min_samples_leaf:
                        # print(f"reached min samples leaf {len}{self.min_samples_leaf}")
                        leaf = True
                    else:
                        break
            else:  # Handling non-binary variables
                for threshold in thresholds:
                    left_indices = X.iloc[:, feature] <= threshold
                    group_indicator = []
                    if len(y[left_indices]) >= self.min_samples_leaf and len(y[~left_indices]) >= self.min_samples_leaf:
                        for is_left in left_indices:
                            if is_left:
                                group_indicator.append(1)  # Samples of left child nodes
                            else:
                                group_indicator.append(0)  # Samples of right child nodes
                        logrank = _calculate_logrank(y, group_indicator)
                        if logrank > best_logrank:
                            best_logrank = logrank
                            best_feature = feature
                            best_threshold = threshold
                    elif len(y[left_indices]) < self.min_samples_leaf and len(y[~left_indices]) < self.min_samples_leaf:
                        leaf = True
        if best_feature is None:
            leaf = True

        return best_feature, best_threshold, leaf

    def _choose_features(self, depth=None):
        if self.probabilities is not None:
            if self.probabilities is not None and type(self.probabilities) != defaultdict:
                selected_features = np.random.choice(self.n_features_, self.max_feature, replace=False,
                                                     p=self.probabilities)
            elif self.probabilities is not None and depth < len(self.probabilities):
                prob = self.probabilities[depth]
                selected_features = np.random.choice(self.n_features_, self.max_feature, replace=False, p=prob)
            else:
                selected_features = np.random.choice(self.n_features_, self.max_feature, replace=False)
        else:
            selected_features = np.random.choice(self.n_features_, self.max_feature, replace=False)
        return selected_features

    def _best_split(self, X, y, node):
        if node.feature is not None:
            selected_features = [node.feature]
        elif self.forest and self.max_feature is not None:
            selected_features = self._choose_features(depth=node.depth)
        # Otherwise, select all features
        else:
            selected_features = range(self.n_features_)

        best_feature, best_threshold, leaf = self.splitter(X, y, selected_features)

        return best_feature, best_threshold, leaf

    def predict(self, X):
        Y_pred = []
        timelines = []
        max_length = 0
        for _, x in X.iterrows():
            y_pred, times = self._predict_survival_function(x, self.tree_)
            Y_pred.append(y_pred)
            timelines.append(times)
            if y_pred.shape[0] > max_length:
                max_length = y_pred.shape[0]
        unique_times = _compute_unique_times(timelines)

        # Pad shorter survival functions
        for i in range(len(Y_pred)):
            if Y_pred[i].shape[0] < max_length:
                last_row = Y_pred[i].iloc[-1]
                padding = pd.DataFrame([last_row] * (max_length - Y_pred[i].shape[0]), columns=Y_pred[i].columns)
                Y_pred[i] = pd.concat([Y_pred[i], padding], ignore_index=True)

        for i in range(len(Y_pred)):
            Y_pred[i] = Y_pred[i].squeeze()

        if self.forest:
            return Y_pred, unique_times

        else:
            self.surv = pd.DataFrame({f"Survival_{i}": Y_pred[i] for i in range(len(Y_pred))},
                                     index=unique_times)
            for col in self.surv.columns:
                self.surv[col] = self.surv[col].ffill()
            return self.surv

    def _predict_survival_function(self, x, node):
        try:
            if node.value is not None:
                return node.value, node.timeline
            feature_value = x.iloc[node.feature]
            if feature_value <= node.threshold:
                return self._predict_survival_function(x, node.left)
            else:
                return self._predict_survival_function(x, node.right)
        except TypeError:
            print(
                f"Cannot find a split for node {node.index}.")
            raise

    def ctd(self, x, y):
        self.predict(x)
        y_test_event = np.array([entry[0] for entry in y], dtype=bool)
        y_test_time = np.array([entry[1] for entry in y])
        ev = EvalSurv(self.surv, y_test_time, y_test_event,
                      censor_surv='km'
                      )  # Kaplan-Meier

        current_ctd = ev.concordance_td('antolini')
        return current_ctd
