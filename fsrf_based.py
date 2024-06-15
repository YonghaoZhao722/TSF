import numpy as np
from sklearn.model_selection import StratifiedKFold
from model.methods import load_rsf_data, grapForest
from global_names import *

if __name__ == "__main__":
    dataset_name = WCH
    X, y = load_rsf_data(dataset_name)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
    average_ctd = []

    for train_index, test_index in kf.split(X, y['status']):
        rsf = grapForest(f"rsf_models/source_forest.pkl")

        rsf.n_estimators = 20
        rsf.min_samples_split = 6
        rsf.min_samples_leaf = 2
        rsf.max_features = 2
        rsf.max_transfer_depth = 1

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rsf.fit(X_train, y_train)
        current_ctd = rsf.ctd(X_test, y_test)
        print(current_ctd)
        average_ctd.append(current_ctd)
    print(f"Average CTD: {np.mean(average_ctd)}")
