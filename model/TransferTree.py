"""Transfer Survival Tree Implementation

This module implements the core survival tree component of Transfer Survival Forest,
with support for transfer learning through tree structure fine-tuning and
depth-wise feature probability guidance.

Key Components:
- Tree: Main survival tree class with transfer learning capabilities
- Node: Tree node implementation with Nelson-Aalen survival estimation
- Transfer Learning: Fine-tuning mechanisms for cross-domain knowledge transfer

The implementation supports both standard survival tree construction and
transfer learning scenarios where source domain knowledge guides target
domain tree building through structure preservation and probability guidance.
"""

from collections import defaultdict

import pandas as pd
from lifelines import NelsonAalenFitter
from pycox.evaluation import EvalSurv
from sksurv.compare import compare_survival
import numpy as np




def _compute_unique_times(timelines):
    """Compute union of unique event times across multiple survival curves.
    
    Aggregates all unique event times from multiple survival function predictions
    to create a unified time grid for ensemble predictions.
    
    Args:
        timelines (list): List of timeline arrays from individual tree predictions
        
    Returns:
        list: Sorted list of unique event times across all timelines
        
    Note:
        Essential for aligning survival functions with different time grids
        in ensemble prediction aggregation.
    """
    unique_times_set = set()
    for timeline in timelines:
        unique_times_set.update(timeline)

    unique_times = list(unique_times_set)
    unique_times.sort()
    return unique_times


def _calculate_logrank(y, group_indicator):
    """Calculate log-rank test statistic for survival curve comparison.
    
    Computes the log-rank test statistic to evaluate the significance of
    differences between survival curves of two groups. Used as the splitting
    criterion in survival tree construction.
    
    Args:
        y (structured array): Survival data with event indicators and times
        group_indicator (list): Binary indicators assigning samples to groups
        
    Returns:
        float: Log-rank chi-square statistic (higher values indicate better splits)
        
    Note:
        The log-rank test is the standard method for comparing survival distributions.
        Returns 0 if computation fails due to insufficient data or other issues.
        
    Splitting Criterion:
        Trees select splits that maximize the log-rank statistic, ensuring
        maximum separation between survival curves of child nodes.
    """
    try:
        chisq, pval = compare_survival(y, group_indicator, return_stats=False)
    except ValueError:
        # Handle cases with insufficient data or computational issues
        chisq = 0

    return chisq


class Node:
    """Survival Tree Node with Nelson-Aalen estimation for terminal predictions.
    
    Represents a single node in a survival tree, capable of serving as either
    an internal splitting node or a terminal prediction node. Terminal nodes
    use Nelson-Aalen estimation to compute cumulative hazard functions.
    
    Class Attributes:
        _index_counter (int): Global counter for unique node indexing
        
    Attributes:
        index (int): Unique identifier for this node
        feature (int): Feature index used for splitting (None for leaf nodes)
        threshold (float): Threshold value for splitting (None for leaf nodes)
        value (pd.Series): Nelson-Aalen cumulative hazard estimate (leaf nodes only)
        left (Node): Left child node
        right (Node): Right child node
        timeline (array): Time points for survival function (leaf nodes only)
        leaf (bool): Whether this is a terminal/leaf node
        depth (int): Depth of this node in the tree
        
    Note:
        The Nelson-Aalen estimator provides non-parametric estimation of
        cumulative hazard functions, which is the standard approach for
        survival tree terminal node predictions.
    """
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
        self.feature = feature  # Feature index for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.value = value  # Nelson-Aalen cumulative hazard (leaf nodes)
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.timeline = timeline  # Time points for survival function
        self.leaf = False  # Leaf node indicator
        self.depth = depth  # Node depth in tree

    def _predict(self, y):
        """Compute terminal node prediction using Nelson-Aalen estimator.
        
        Fits a Nelson-Aalen estimator to the survival data in this terminal node
        to estimate the cumulative hazard function. This provides the node's
        prediction for any samples that reach this leaf.
        
        Args:
            y (structured array): Survival data with 'Survival.months' and 'status'
            
        Side Effects:
            - Sets self.value to cumulative hazard function
            - Sets self.timeline to corresponding time points
            - Marks node as leaf and removes child references
            
        Nelson-Aalen Estimation:
            The Nelson-Aalen estimator is the standard non-parametric method
            for estimating cumulative hazard functions in survival analysis.
            It handles both censored and uncensored observations appropriately.
        """
        naf = NelsonAalenFitter()
        naf.fit(y['Survival.months'], event_observed=y['status'])
        self.value = naf.cumulative_hazard_  # Cumulative hazard function
        self.timeline = naf.timeline  # Corresponding time points

        # Mark as terminal node
        self.leaf = True
        self.left = None
        self.right = None

def bootstrap_data(X, y, random_state):
    """Bootstrap sampling for individual tree training in transfer learning context.
    
    Performs bootstrap resampling specifically for fine-tuning pre-existing trees
    in transfer learning scenarios. Ensures bootstrap samples contain both
    event types for valid survival analysis.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (structured array): Survival data with 'status' field
        random_state (int): Random seed for reproducible sampling
        
    Returns:
        tuple: (X_bootstrapped, y_bootstrapped) - Bootstrap samples
        
    Note:
        This function is used during tree fine-tuning in TSF-Tk method,
        providing diverse data subsets for robust transfer learning.
    """
    random_state = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    indices = random_state.choice(n_samples, n_samples, replace=True)
    indices = np.unique(indices)
    X_bootstrapped = X.iloc[indices]
    y_bootstrapped = y[indices]
    
    # Ensure both censored and uncensored events are present
    while len(np.unique(y_bootstrapped['status'])) < 2:
        indices = random_state.choice(n_samples, n_samples, replace=True)
        indices = np.unique(indices)
        X_bootstrapped = X.iloc[indices]
        y_bootstrapped = y[indices]
    return X_bootstrapped, y_bootstrapped


class Tree:
    """Transfer Survival Tree with Cross-Domain Knowledge Transfer.
    
    Implements a survival tree capable of both standard construction and
    transfer learning from pre-existing source domain trees. Supports two
    main transfer learning approaches:
    
    1. TSF-Tk (Tree Structure Frequency): Fine-tunes existing tree structures
       from source domain by preserving splitting features and recomputing
       thresholds on target data
    
    2. DP (Depth-wise Probability): Uses source domain feature usage patterns
       to guide feature selection during target domain tree construction
    
    Parameters:
        max_depth (int, optional): Maximum tree depth. None for unlimited depth
        min_samples_split (int, default=6): Minimum samples required to split a node
        min_samples_leaf (int, default=0): Minimum samples required in leaf nodes
        random_state (int, default=1234): Random seed for reproducibility
        max_feature (int, optional): Number of features to consider for splits
        forest (bool, default=False): Whether this tree is part of a forest
        max_transfer_depth (int, optional): Maximum depth for transfer learning.
                                          Beyond this depth, standard growth occurs
    
    Attributes:
        n_features_ (int): Number of features in training data
        tree_ (Node): Root node of the fitted tree
        nodes (list): List of all nodes in the tree
        probabilities (dict): Depth-wise feature probabilities for DP method
        surv (pd.DataFrame): Survival function predictions (for single trees)
        
    Transfer Learning Concepts:
        - Transfer Depth (k): Controls how many layers inherit source structure
        - Fine-tuning: Adapts source trees to target domain while preserving structure
        - Feature Guidance: Uses source patterns to inform target feature selection
    """
    
    def __init__(self, max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=0,
                 random_state=1234,
                 max_feature=None,
                 forest=False,
                 max_transfer_depth=None, ):
        # Core tree structure
        self.n_features_ = None  # Number of features (set during fitting)
        self.tree_ = None  # Root node of the tree
        
        # Tree construction parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_feature = max_feature
        self.random_state = random_state
        self.forest = forest  # Whether part of a Random Forest
        self.nodes = []  # List of all nodes for tree traversal
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Transfer learning parameters
        self.max_transfer_depth = max_transfer_depth  # Depth limit for knowledge transfer
        self.probabilities = None  # Feature probabilities for DP method

    def fit(self, X, y):
        """Fit the survival tree using standard construction or transfer learning.
        
        Determines the training approach based on whether a pre-existing tree
        structure is available:
        
        1. Standard Construction: Builds tree from scratch using log-rank splitting
        2. Transfer Learning: Fine-tunes existing tree structure on target data
        
        Args:
            X (pd.DataFrame): Feature matrix with shape (n_samples, n_features)
            y (structured array): Survival data with 'status' and time information
            
        Returns:
            Tree: Self for method chaining
            
        Transfer Learning Flow:
            - Uses bootstrap sampling for robust fine-tuning
            - Calls self.retrain() which implements TSF-Tk fine-tuning logic
            - Preserves source tree structure while adapting to target distribution
        """
        self.n_features_ = X.shape[1]
        
        if self.tree_ is None:
            # Standard construction: build tree from scratch
            self.tree_ = self._grow_tree(X, y)
            Node._index_counter = 0
        else:
            # Transfer learning: fine-tune existing tree structure
            X, y = bootstrap_data(X, y, self.random_state)
            self.retrain(X, y)  # Implements TSF-Tk fine-tuning
            Node._index_counter = 0

        return self

    def _grow_tree(self, X, y, depth=0, ):
        """Recursively construct survival tree using log-rank splitting criterion.
        
        Builds a survival tree from scratch by recursively finding optimal splits
        that maximize separation between survival curves using log-rank test statistic.
        
        Args:
            X (pd.DataFrame): Feature matrix for current node samples
            y (structured array): Survival data for current node samples
            depth (int, default=0): Current depth in the tree
            
        Returns:
            Node: Root node of the constructed subtree
            
        Stopping Criteria:
            - Maximum depth reached
            - Insufficient samples for splitting (min_samples_split)
            - No valid splits found
            - Insufficient samples for child nodes (min_samples_leaf)
            
        Splitting Process:
            1. Evaluate all possible feature-threshold combinations
            2. Select split maximizing log-rank test statistic
            3. Handle both binary and continuous features appropriately
            4. Recursively build left and right subtrees
        """
        n_samples = len(y)
        node = Node(depth=depth)
        self.nodes.append(node)
        
        # Check stopping criteria
        if depth == self.max_depth or n_samples < self.min_samples_split:
            node._predict(y)  # Create terminal node with Nelson-Aalen prediction
            return node

        # Find optimal split using log-rank criterion
        feature, threshold, leaf = self._best_split(X, y, node=node)
        
        if leaf:
            node._predict(y)
            return node
            
        if feature is not None:
            # Handle binary vs continuous features
            if len(np.unique(X.iloc[:, feature])) == 2:  # Binary feature
                indices_left = X.iloc[:, feature] == 0
            else:  # Continuous feature
                indices_left = X.iloc[:, feature] <= threshold
                
            X_left, y_left = X.loc[indices_left], y[indices_left]
            X_right, y_right = X.loc[~indices_left], y[~indices_left]
            
            # Ensure sufficient samples in child nodes
            if len(y_left) >= self.min_samples_leaf and len(y_right) >= self.min_samples_leaf:
                node.feature = feature
                node.threshold = threshold
                # Recursively build subtrees
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
            else:
                node._predict(y)  # Create leaf if child nodes would be too small
        else:
            node._predict(y)  # No valid split found
            
        return node

    def retrain(self, X, y):
        """Fine-tune existing tree structure for transfer learning (TSF-Tk method).
        
        Implements the core transfer learning mechanism by fine-tuning a pre-existing
        tree structure on new target domain data. This is the main entry point for
        the TSF-Tk (Tree Structure Frequency) transfer learning approach.
        
        Args:
            X (pd.DataFrame): Target domain feature matrix
            y (structured array): Target domain survival data
            
        Raises:
            ValueError: If no pre-existing tree structure is available
            
        Transfer Learning Process:
            - Preserves source domain tree structure (splitting features)
            - Recomputes optimal thresholds using target domain data
            - Selectively grows new nodes beyond transfer depth
            - Maintains tree topology while adapting to target distribution
        """
        if self.tree_ is not None:
            self.tree_ = self._retrain_tree(self.tree_, X, y, depth=0)
        else:
            raise ValueError("The tree has not been trained yet. Please call fit() method first.")

    def _retrain_tree(self, node, X, y, depth):
        """Core transfer learning algorithm for fine-tuning tree structures.
        
        Implements the TSF-Tk fine-tuning mechanism that selectively preserves
        source domain structure while adapting to target domain data. The transfer
        depth parameter controls the balance between structure preservation and
        adaptation flexibility.
        
        Args:
            node (Node): Current node to fine-tune
            X (pd.DataFrame): Target domain features for current node
            y (structured array): Target domain survival data for current node
            depth (int): Current depth in the tree
            
        Returns:
            Node: Fine-tuned node with updated structure
            
        Transfer Learning Logic:
            - depth < max_transfer_depth: Preserve splitting feature, recompute threshold
            - depth >= max_transfer_depth: Standard tree growth (no structure preservation)
            
        Key Concepts:
            1. Structure Preservation: Maintains source domain splitting features
            2. Threshold Adaptation: Recomputes thresholds using target data
            3. Progressive Freedom: Allows more flexibility at greater depths
            4. Robustness Checks: Ensures valid splits with sufficient data
        """
        n_samples = len(y)
        
        # Check stopping criteria
        if depth == self.max_depth or n_samples < self.min_samples_split:
            node._predict(y)
            return node

        # Transfer learning: preserve structure within transfer depth
        if node.feature is not None:
            if self.max_transfer_depth is None or depth < self.max_transfer_depth:
                # Preserve splitting feature, recompute threshold
                best_feature, best_threshold, leaf = self._best_split(X, y, node=node)

                if leaf:
                    # Convert to terminal node if no valid split
                    node._predict(y)
                    node.left = None
                    node.right = None
                    return node
                else:
                    # Update threshold while preserving feature
                    node.threshold = best_threshold

        # Process child nodes if not a leaf
        if not node.leaf:
            # Handle binary vs continuous features for data splitting
            if len(np.unique(X.iloc[:, node.feature])) == 2:  # Binary feature
                left_indices = X.iloc[:, node.feature] == 0
                right_indices = X.iloc[:, node.feature] == 1
            else:  # Continuous feature
                left_indices = X.iloc[:, node.feature] < node.threshold
                right_indices = X.iloc[:, node.feature] >= node.threshold

            X_left, y_left = X.loc[left_indices], y[left_indices]
            X_right, y_right = X.loc[right_indices], y[right_indices]
            
            # Robustness checks: ensure valid data splits
            if (np.unique(left_indices).shape[0] == 1) or (np.unique(right_indices).shape[0] == 1):
                node._predict(y)
                return node

            # Check for sufficient event diversity (avoid all-censored nodes)
            if len(np.unique(y_left['status'])) == 1 or len(np.unique(y_right['status'])) == 1:
                node._predict(y)
                return node

            # Recursive processing based on transfer depth
            if self.max_transfer_depth is None or depth < self.max_transfer_depth:
                # Within transfer depth: continue fine-tuning
                node.left = self._retrain_tree(node.left, X_left, y_left, depth + 1)
            else:
                # Beyond transfer depth: standard tree growth
                node.left = self._grow_tree(X_left, y_left, depth + 1)

            if self.max_transfer_depth is None or depth < self.max_transfer_depth:
                # Within transfer depth: continue fine-tuning
                node.right = self._retrain_tree(node.right, X_right, y_right, depth + 1)
            else:
                # Beyond transfer depth: standard tree growth
                node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node

    def splitter(self, X, y, selected_features):
        """Find optimal split using log-rank test criterion for survival trees.
        
        Evaluates all possible feature-threshold combinations to find the split
        that maximizes separation between survival curves using log-rank statistic.
        
        Args:
            X (pd.DataFrame): Feature matrix for current node
            y (structured array): Survival data for current node
            selected_features (list): Feature indices to consider for splitting
            
        Returns:
            tuple: (best_feature, best_threshold, leaf)
                - best_feature: Index of optimal splitting feature (None if leaf)
                - best_threshold: Optimal threshold value (None if leaf)
                - leaf: Boolean indicating if node should be terminal
                
        Splitting Strategy:
            1. Binary Features: Test both possible thresholds (0 and 1)
            2. Continuous Features: Test all unique values as thresholds
            3. Log-rank Criterion: Select split maximizing survival curve separation
            4. Sample Size Constraints: Ensure sufficient samples in child nodes
            
        Note:
            The log-rank test statistic measures the significance of differences
            between survival distributions, making it ideal for survival tree splitting.
        """
        leaf = False
        best_logrank = 0
        best_feature, best_threshold = None, None
        
        for feature in selected_features:
            thresholds = np.unique(X.iloc[:, feature])
            
            if len(thresholds) == 2:  # Binary feature handling
                for threshold in thresholds:
                    left_indices = X.iloc[:, feature] == threshold
                    group_indicator = []
                    
                    # Check minimum sample requirements
                    if len(y[left_indices]) >= self.min_samples_leaf and len(y[~left_indices]) >= self.min_samples_leaf:
                        # Create group indicators for log-rank test
                        for is_left in left_indices:
                            if is_left:
                                group_indicator.append(1)  # Left child group
                            else:
                                group_indicator.append(0)  # Right child group
                                
                        # Compute log-rank statistic
                        logrank = _calculate_logrank(y, group_indicator)
                        if logrank > best_logrank:
                            best_logrank = logrank
                            best_feature = feature
                            best_threshold = threshold
                            
                    elif len(y[left_indices]) < self.min_samples_leaf and len(y[~left_indices]) < self.min_samples_leaf:
                        leaf = True  # Both child nodes would be too small
                    else:
                        break  # One child too small, try next threshold
                        
            else:  # Continuous feature handling
                for threshold in thresholds:
                    left_indices = X.iloc[:, feature] <= threshold
                    group_indicator = []
                    
                    # Check minimum sample requirements
                    if len(y[left_indices]) >= self.min_samples_leaf and len(y[~left_indices]) >= self.min_samples_leaf:
                        # Create group indicators for log-rank test
                        for is_left in left_indices:
                            if is_left:
                                group_indicator.append(1)  # Left child group
                            else:
                                group_indicator.append(0)  # Right child group
                                
                        # Compute log-rank statistic
                        logrank = _calculate_logrank(y, group_indicator)
                        if logrank > best_logrank:
                            best_logrank = logrank
                            best_feature = feature
                            best_threshold = threshold
                            
                    elif len(y[left_indices]) < self.min_samples_leaf and len(y[~left_indices]) < self.min_samples_leaf:
                        leaf = True  # Both child nodes would be too small
                        
        # Force leaf creation if no valid split found
        if best_feature is None:
            leaf = True

        return best_feature, best_threshold, leaf

    def _choose_features(self, depth=None):
        """Select features for splitting using DP method guidance or random selection.
        
        Implements feature selection with optional depth-wise probability guidance
        from source domain knowledge (DP method). This allows transfer learning
        by biasing feature selection toward patterns observed in source domain.
        
        Args:
            depth (int, optional): Current tree depth for depth-specific probabilities
            
        Returns:
            np.ndarray: Array of selected feature indices
            
        Feature Selection Strategies:
            1. DP Method: Use depth-specific probabilities from source domain
            2. Global Probabilities: Use uniform probabilities across depths
            3. Random Selection: Uniform random selection (standard Random Forest)
            
        DP Method Implementation:
            - Loads pre-computed probabilities P_k(f) for feature f at depth k
            - Guides target domain feature selection using source patterns
            - Falls back to random selection if no probabilities available
        """
        if self.probabilities is not None:
            # Check if probabilities are depth-specific (DP method)
            if self.probabilities is not None and type(self.probabilities) != defaultdict:
                # Use global probabilities (not depth-specific)
                selected_features = np.random.choice(self.n_features_, self.max_feature, replace=False,
                                                     p=self.probabilities)
            elif self.probabilities is not None and depth < len(self.probabilities):
                # Use depth-specific probabilities (DP method)
                prob = self.probabilities[depth]
                selected_features = np.random.choice(self.n_features_, self.max_feature, replace=False, p=prob)
            else:
                # Fallback to random selection if depth exceeds probability coverage
                selected_features = np.random.choice(self.n_features_, self.max_feature, replace=False)
        else:
            # Standard random feature selection (no transfer learning)
            selected_features = np.random.choice(self.n_features_, self.max_feature, replace=False)
            
        return selected_features

    def _best_split(self, X, y, node):
        """Find the best split for a given node using appropriate feature selection.
        
        Coordinates feature selection and split evaluation, handling different
        scenarios including transfer learning fine-tuning and standard construction.
        
        Args:
            X (pd.DataFrame): Feature matrix for current node
            y (structured array): Survival data for current node
            node (Node): Current node being processed
            
        Returns:
            tuple: (best_feature, best_threshold, leaf) from splitter method
            
        Feature Selection Logic:
            1. Transfer Learning: Use pre-existing feature if available (fine-tuning)
            2. Forest Mode: Random feature subset with optional DP guidance
            3. Single Tree: Consider all available features
            
        Note:
            This method serves as the interface between feature selection strategies
            and the actual split evaluation performed by the splitter method.
        """
        if node.feature is not None:
            # Transfer learning: preserve existing splitting feature
            selected_features = [node.feature]
        elif self.forest and self.max_feature is not None:
            # Forest mode: random feature subset (with optional DP guidance)
            selected_features = self._choose_features(depth=node.depth)
        else:
            # Single tree mode: consider all features
            selected_features = range(self.n_features_)

        # Find optimal split among selected features
        best_feature, best_threshold, leaf = self.splitter(X, y, selected_features)

        return best_feature, best_threshold, leaf

    def predict(self, X):
        """Generate survival function predictions for input samples.
        
        Produces cumulative hazard function estimates for each sample by traversing
        the fitted tree structure and aggregating terminal node predictions.
        
        Args:
            X (pd.DataFrame): Feature matrix for prediction samples
            
        Returns:
            For forest mode: tuple (Y_pred, unique_times)
                - Y_pred: List of individual survival predictions
                - unique_times: Union of all event times
            For single tree: pd.DataFrame with survival functions
            
        Prediction Process:
            1. Traverse tree for each sample to reach terminal node
            2. Extract Nelson-Aalen cumulative hazard from terminal node
            3. Align predictions across different time grids using padding
            4. Return appropriate format based on usage context (forest vs single tree)
            
        Note:
            Forward filling ensures step function properties of survival curves
            are maintained throughout the prediction process.
        """
        Y_pred = []
        timelines = []
        max_length = 0
        
        # Generate individual predictions
        for _, x in X.iterrows():
            y_pred, times = self._predict_survival_function(x, self.tree_)
            Y_pred.append(y_pred)
            timelines.append(times)
            if y_pred.shape[0] > max_length:
                max_length = y_pred.shape[0]
                
        unique_times = _compute_unique_times(timelines)

        # Align predictions by padding shorter survival functions
        for i in range(len(Y_pred)):
            if Y_pred[i].shape[0] < max_length:
                last_row = Y_pred[i].iloc[-1]
                padding = pd.DataFrame([last_row] * (max_length - Y_pred[i].shape[0]), columns=Y_pred[i].columns)
                Y_pred[i] = pd.concat([Y_pred[i], padding], ignore_index=True)

        # Convert to series format
        for i in range(len(Y_pred)):
            Y_pred[i] = Y_pred[i].squeeze()

        if self.forest:
            # Return raw predictions for ensemble aggregation
            return Y_pred, unique_times
        else:
            # Format for single tree usage
            self.surv = pd.DataFrame({f"Survival_{i}": Y_pred[i] for i in range(len(Y_pred))},
                                     index=unique_times)
            # Forward fill to maintain step function properties
            for col in self.surv.columns:
                self.surv[col] = self.surv[col].ffill()
            return self.surv

    def _predict_survival_function(self, x, node):
        """Recursively traverse tree to generate survival prediction for single sample.
        
        Navigates the tree structure following splitting rules until reaching a
        terminal node, then returns the Nelson-Aalen cumulative hazard estimate.
        
        Args:
            x (pd.Series): Feature vector for single sample
            node (Node): Current node in tree traversal
            
        Returns:
            tuple: (cumulative_hazard, timeline)
                - cumulative_hazard: Nelson-Aalen estimate from terminal node
                - timeline: Corresponding time points for the estimate
                
        Traversal Logic:
            - Terminal nodes: Return stored Nelson-Aalen prediction
            - Internal nodes: Compare feature value to threshold and recurse
            - Left child: feature_value <= threshold
            - Right child: feature_value > threshold
            
        Note:
            Handles both binary and continuous features through threshold comparison.
            Raises TypeError if tree structure is inconsistent or corrupted.
        """
        try:
            if node.value is not None:
                # Terminal node: return Nelson-Aalen survival prediction
                return node.value, node.timeline
                
            # Internal node: traverse based on splitting rule
            feature_value = x.iloc[node.feature]
            if feature_value <= node.threshold:
                return self._predict_survival_function(x, node.left)
            else:
                return self._predict_survival_function(x, node.right)
                
        except TypeError:
            print(f"Cannot find a split for node {node.index}.")
            raise

    def ctd(self, x, y):
        """Compute Concordance Index (C-index) for single tree evaluation.
        
        Evaluates the tree's predictive performance using Antolini's time-dependent
        concordance measure, which properly handles censoring in survival analysis.
        
        Args:
            x (pd.DataFrame): Test feature matrix
            y (structured array): Test survival data with event indicators and times
            
        Returns:
            float: Concordance index (higher values indicate better performance)
            
        Evaluation Process:
            1. Generate survival predictions using self.predict()
            2. Extract event indicators and survival times from test data
            3. Compute concordance using Kaplan-Meier censoring adjustment
            4. Return Antolini's time-dependent concordance measure
            
        Note:
            The concordance index measures the proportion of correctly ordered
            pairs in terms of survival times, accounting for censoring. Values
            range from 0.5 (random) to 1.0 (perfect concordance).
        """
        self.predict(x)
        
        # Extract event indicators and survival times
        y_test_event = np.array([entry[0] for entry in y], dtype=bool)
        y_test_time = np.array([entry[1] for entry in y])
        
        # Evaluate survival predictions with proper censoring handling
        ev = EvalSurv(self.surv, y_test_time, y_test_event,
                      censor_surv='km')  # Kaplan-Meier censoring adjustment

        # Compute Antolini's time-dependent concordance
        current_ctd = ev.concordance_td('antolini')
        return current_ctd
