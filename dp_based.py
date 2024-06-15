import numpy as np
from sklearn.model_selection import StratifiedKFold
from model.TransferSurvivalForest import Forest,get_probabilities
from model.methods import load_rsf_data
from global_names import *



if __name__ == "__main__":
    features_order = ['gender_code', 'age', 'tumor.size', 'grade_code', 'T', 'lymphcat', 'PNI', 'cea_positive']
    dataset_name = WCH
    X, y = load_rsf_data(dataset_name)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

    average_ctd = []
    for train_index, test_index in kf.split(X, y['status']):
        rsf = Forest(n_estimators=50,
                     min_samples_leaf=6,
                     min_samples_split=13,
                     max_features=3)
        rsf.probabilities = get_probabilities(features_order)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rsf.fit(X_train, y_train)
        current_ctd = rsf.ctd(X_test, y_test)

        print(current_ctd)
        average_ctd.append(current_ctd)
    print(f"Average CTD: {np.mean(average_ctd)}")

