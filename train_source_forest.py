import os
from model.TransferSurvivalForest import Forest
from model.methods import load_rsf_data, storeForest, set_all_seeds
from global_names import *
from calculate_dp import calculate_and_write_total_feature_probability
def train_source_forest():
    # Set all random seeds for reproducibility
    set_all_seeds(1234)
    
    dataset_name = SEER
    X, y = load_rsf_data(dataset_name)

    rsf = Forest(n_estimators=300,
                 min_samples_split=20,
                 min_samples_leaf=10,
                 random_state=1234,
                 max_features=3,
                 deterministic=True,)


    rsf.fit(X, y)
    path = "rsf_models/"
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "source_forest.pkl")
    storeForest(rsf, model_path)
    print(f"RSF model saved at {model_path}")
    return model_path


if __name__=="__main__":
    # Set all random seeds for reproducibility
    set_all_seeds(1234)
    
    source_forest_path = train_source_forest()

    output_csv_file = "dp.csv"
    feature_mapping = {i: feature for i, feature in enumerate(
        ['gender_code', 'age', 'tumor.size', 'grade_code', 'T', 'lymphcat', 'PNI', 'cea_positive'])}
    calculate_and_write_total_feature_probability(source_forest_path, output_csv_file,feature_mapping)
