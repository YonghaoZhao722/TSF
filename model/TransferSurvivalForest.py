"""Transfer Survival Forest Implementation

This module implements Transfer Survival Forest (TSF), a novel approach for survival analysis
that leverages knowledge transfer from source domains to improve prediction performance
on target domains with limited data.

Key Methods:
- TSF-Tk: Tree Structure Frequency method that transfers tree structures from source to target
- DP: Depth-wise Feature Probability method that guides feature selection using source knowledge

References:
    Based on Transfer Survival Forest methodology for cross-domain survival analysis.
"""

import csv
from collections import defaultdict
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from joblib import Parallel, delayed
from model.TransferTree import Tree
import os
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/temp/joblib'  # Compatibility settings for parallel computing



def bootstrap_data(X, y, tree_random):
    """Bootstrap sampling for Random Survival Forest.
    
    Performs bootstrap resampling (sampling with replacement) to create diverse
    training subsets for each tree in the forest. Ensures that each bootstrap
    sample contains both censored and uncensored events.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (structured array): Survival data with 'status' (event indicator) and time
        tree_random (np.random.RandomState): Random state for reproducible sampling
        
    Returns:
        tuple: (X_bootstrapped, y_bootstrapped) - Bootstrap samples
        
    Note:
        Continues resampling until the bootstrap sample contains both event types
        (censored and uncensored) to ensure valid survival analysis.
    """
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
    """Load depth-wise feature probabilities for DP (Depth-wise Probability) method.
    
    Reads pre-computed feature selection probabilities from source forest analysis.
    These probabilities P_k(f) represent how frequently each feature f appears
    at depth k in the source domain trees.
    
    Args:
        features_order (list): Ordered list of feature names to ensure consistent indexing
        
    Returns:
        dict: Mapping from depth to probability distribution over features
              Format: {depth: [prob_feature1, prob_feature2, ...]}
              
    Note:
        - Reads from 'dp.csv' containing depth, feature, probability triplets
        - Normalizes probabilities to sum to 1.0 at each depth
        - Assigns minimum probability to missing features to avoid zero probabilities
        
    DP Method:
        This implements the knowledge transfer component where source domain
        feature usage patterns guide target domain tree construction.
    """
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
        # Assign minimum probability to features not seen at this depth
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
    """Convert survival predictions to step function format.
    
    Transforms individual tree predictions into a consistent step function
    representation for ensemble aggregation.
    
    Args:
        Y_pred: Survival function predictions from individual tree
        
    Returns:
        pd.DataFrame: Step function representation with forward-filled values
        
    Note:
        Forward filling ensures that survival probabilities are monotonically
        non-increasing as required by survival analysis theory.
    """
    Y_pred_df = pd.DataFrame(Y_pred).transpose()
    Y_pred_df = Y_pred_df.sort_index()
    Y_pred_df = Y_pred_df.ffill()
    return Y_pred_df


class Forest:
    """Transfer Survival Forest - Random Survival Forest with Transfer Learning.
    
    Implements a Random Survival Forest that can leverage knowledge from source domains
    through two main transfer learning approaches:
    
    1. TSF-Tk (Tree Structure Frequency): Transfers tree structures from source forest
       and fine-tunes them on target data
    2. DP (Depth-wise Probability): Uses source domain feature usage patterns to
       guide feature selection in target domain trees
    
    Parameters:
        n_estimators (int, default=100): Number of trees in the forest
        max_depth (int, optional): Maximum depth of trees. None for unlimited depth
        min_samples_split (int, default=6): Minimum samples required to split a node
        min_samples_leaf (int, default=0): Minimum samples required in leaf nodes
        max_features (int, optional): Number of features to consider for best split
        random_state (int, default=1234): Random seed for reproducibility
        max_transfer_depth (int, optional): Maximum depth for transfer learning.
                                          Beyond this depth, trees grow normally
        deterministic (bool, default=True): Whether to use deterministic execution
                                          (single-threaded) for reproducible results
    
    Attributes:
        unique_times: Union of all unique event times from forest predictions
        estimators: List of fitted Tree objects
        probabilities: Depth-wise feature probabilities for DP method
        surv: Aggregated survival function predictions
    
    Methods:
        fit(X, y): Train the forest using transfer learning or from scratch
        predict(X): Generate ensemble survival predictions
        ctd(x, y): Compute concordance index for evaluation
        load_softmax_probabilities(features_order): Load DP method probabilities
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=6,
                 min_samples_leaf=0, max_features=None, random_state=1234,
                 max_transfer_depth=None, deterministic=True):
        # Core forest parameters
        self.unique_times = None  # Union of all unique event times from predictions
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = np.random.RandomState(random_state)
        
        # Transfer learning components
        self.estimators = None  # Pre-trained trees for TSF-Tk method
        self.bootstrap = True
        self.max_transfer_depth = max_transfer_depth
        self.probabilities = None
        self.deterministic = deterministic

    def fit_one_tree(self, args):
        """Train a single survival tree with optional transfer learning.
        
        Creates and trains one tree in the forest using bootstrap sampling.
        Incorporates DP method probabilities if available for guided feature selection.
        
        Args:
            args (tuple): (X, y, (idx, tree_random_state)) containing:
                - X: Feature matrix
                - y: Survival data
                - idx: Tree index
                - tree_random_state: Random seed for this tree
                
        Returns:
            Tree: Fitted survival tree
            
        Note:
            This method supports both standard Random Survival Forest training
            and DP method transfer learning through feature probability guidance.
        """
        X, y, (idx, tree_random_state) = args
        tree_random = np.random.RandomState(tree_random_state)
        X_bootstrapped, y_bootstrapped = bootstrap_data(X, y, tree_random)
        
        # Create tree with transfer learning parameters
        estimator = Tree(max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         min_samples_leaf=self.min_samples_leaf,
                         max_feature=self.max_features,
                         random_state=tree_random_state,
                         max_transfer_depth=self.max_transfer_depth,
                         forest=True,
                         )
        
        # Apply DP method: transfer feature probabilities from source domain
        if self.probabilities is not None:
            estimator.probabilities = self.probabilities
            
        estimator.fit(X_bootstrapped, y_bootstrapped)
        return estimator

    def fit(self, X, y):
        """Train the Transfer Survival Forest.
        
        Implements two training modes:
        1. TSF-Tk Method: Fine-tune pre-existing tree structures from source domain
        2. Standard Training: Build new forest from scratch (with optional DP guidance)
        
        Args:
            X (pd.DataFrame): Feature matrix with shape (n_samples, n_features)
            y (structured array): Survival data with 'status' and time fields
            
        TSF-Tk Transfer Learning Process:
            - Selects prototype tree structures from source domain (self.estimators)
            - Updates tree hyperparameters for target domain
            - Fine-tunes selected trees using target data via retrain mechanism
            - Preserves beneficial source structures while adapting to target distribution
            
        Standard Training Process:
            - Creates new trees from scratch using bootstrap sampling
            - Optionally applies DP method for guided feature selection
            - Builds complete Random Survival Forest
        """
        if self.estimators:
            # TSF-Tk Method: Transfer and fine-tune source tree structures
            selected_estimators = []

            # Sample prototype trees from source domain, excluding trivial trees
            while len(selected_estimators) < self.n_estimators:
                estimator = self.random_state.choice(self.estimators)

                # Skip trees with only one node (no useful structure to transfer)
                if len(estimator.nodes) == 1:
                    continue

                selected_estimators.append(estimator)

            selected_estimators = np.array(selected_estimators)
            
            # Update hyperparameters for target domain fine-tuning
            for estimators in selected_estimators:
                estimators.min_samples_split = self.min_samples_split
                estimators.min_samples_leaf = self.min_samples_leaf
                estimators.max_feature = self.max_features
                estimators.max_transfer_depth = self.max_transfer_depth

            self.estimators = []
            
            # Fine-tune selected trees on target data
            if self.deterministic:
                # Use single-threaded execution for deterministic results
                self.estimators = []
                for estimator in selected_estimators:
                    estimator.fit(X, y)  # Triggers _retrain_tree for fine-tuning
                    self.estimators.append(estimator)
            else:
                self.estimators = (
                    Parallel(n_jobs=-1)(
                        delayed(estimator.fit)(X, y) for estimator in selected_estimators
                    ))
        else:
            # Standard Training: Build new forest from scratch
            self.estimators = []
            tree_random_states = [(idx, self.random_state.randint(0, 10000)) for idx in range(self.n_estimators)]
            
            if self.deterministic:
                # Use single-threaded execution for deterministic results
                self.estimators = []
                for rs in tree_random_states:
                    estimator = self.fit_one_tree((X, y, rs))
                    self.estimators.append(estimator)
            else:
                self.estimators = (
                    Parallel(n_jobs=-1)(
                        delayed(self.fit_one_tree)((X, y, rs)) for rs in tree_random_states
                    ))

    def predict(self, X):
        """Generate ensemble survival predictions.
        
        Combines predictions from all trees in the forest to produce robust
        survival function estimates through ensemble averaging.
        
        Args:
            X (pd.DataFrame): Feature matrix for prediction samples
            
        Returns:
            None: Predictions stored in self.surv attribute
            
        Ensemble Process:
            1. Collect predictions from all valid trees (excluding trivial single-node trees)
            2. Convert each tree's prediction to step function format
            3. Align time indices across all predictions using reindexing
            4. Average predictions across trees for final ensemble estimate
            
        Note:
            The ensemble approach reduces variance and improves generalization
            compared to individual tree predictions. Self.surv contains the
            final aggregated survival functions.
        """
        predictions = None
        all_unique_value = set()
        
        for estimator in self.estimators:
            # Skip trees with only one node (no predictive structure)
            if len(estimator.nodes) == 1:
                continue
                
            Y_pred, unique_times = estimator.predict(X)
            all_unique_value.update(unique_times)
            Y_pred_df = get_step_function(Y_pred)

            if predictions is None:
                # Initialize with first valid prediction
                predictions = Y_pred_df.copy()
            else:
                # Align time indices and aggregate predictions
                # Use reindex and ffill to handle different time grids across trees
                predictions = predictions.reindex(index=predictions.index.union(Y_pred_df.index)).ffill().add(
                    Y_pred_df.reindex(index=predictions.index.union(Y_pred_df.index)).ffill())

        # Compute ensemble average
        predictions /= len(self.estimators)
        self.unique_times = all_unique_value
        self.surv = predictions

    def ctd(self, x, y):
        """Compute Concordance Index (C-index) for model evaluation.
        
        Evaluates the model's ability to correctly rank survival times using
        Antolini's time-dependent concordance measure.
        
        Args:
            x (pd.DataFrame): Test feature matrix
            y (structured array): Test survival data with events and times
            
        Returns:
            float: Concordance index (higher values indicate better performance)
            
        Note:
            Uses Antolini's concordance measure which properly handles censoring
            and provides time-dependent evaluation. Converts cumulative hazard
            to survival probabilities via exponential transformation.
        """
        self.predict(x)
        survival_df = np.exp(-self.surv)  # Convert cumulative hazard to survival probability
        y_test_event = np.array([entry[0] for entry in y], dtype=bool)
        y_test_time = np.array([entry[1] for entry in y])
        ev = EvalSurv(survival_df, y_test_time, y_test_event)
        current_ctd = ev.concordance_td('antolini')
        return current_ctd


    def load_softmax_probabilities(self, features_order):
        """Load depth-wise feature probabilities for DP method transfer learning.
        
        Loads pre-computed feature selection probabilities from source domain analysis.
        These probabilities guide feature selection during target domain tree construction,
        implementing the DP (Depth-wise Probability) transfer learning approach.
        
        Args:
            features_order (list): Ordered list of feature names for consistent indexing
            
        Note:
            This method enables knowledge transfer by incorporating source domain
            patterns into target domain tree building. The probabilities are used
            during the _choose_features method in individual trees.
        """
        self.probabilities = get_probabilities(features_order)