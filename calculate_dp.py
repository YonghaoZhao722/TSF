from collections import defaultdict
from model.methods import grapForest
import csv

def save_node_features(rsf):
    tree_features = []
    for estimator in rsf.estimators:
        node_features = {node.index: (node.feature, node.depth) for node in estimator.nodes if node.feature is not None}
        tree_features.append(node_features)
    return tree_features

def calculate_feature_probability(tree_features,feature_mapping):
    depth_feature_counts = defaultdict(lambda: defaultdict(int))
    depth_counts = defaultdict(int)

    for tree in tree_features:
        for feature, depth in tree.values():
            depth_feature_counts[depth][feature] += 1
            depth_counts[depth] += 1

    depthwise_probabilities = defaultdict(dict)
    for depth, feature_counts in depth_feature_counts.items():
        total_depth_count = depth_counts[depth]
        for feature, count in feature_counts.items():
            feature_name = feature_mapping.get(feature, f"Unknown feature {feature}")
            depthwise_probabilities[depth][feature_name] = count / total_depth_count

    return depthwise_probabilities

def calculate_and_write_total_feature_probability(file_path, output_csv_file, feature_mapping):
    rsf = grapForest(file_path)
    tree_features = save_node_features(rsf)
    depth_feature_probabilities = calculate_feature_probability(tree_features,feature_mapping)

    total_depth_feature_counts = defaultdict(lambda: defaultdict(float))
    total_depth_counts = defaultdict(int)

    for depth, probabilities in depth_feature_probabilities.items():
        total_depth_count = sum(probabilities.values())
        for feature, probability in probabilities.items():
            total_depth_feature_counts[depth][feature] += probability / total_depth_count
        total_depth_counts[depth] += 1

    total_feature_probabilities = defaultdict(dict)
    for depth, feature_counts in total_depth_feature_counts.items():
        total_depth_count = total_depth_counts[depth]
        for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
            total_feature_probabilities[depth][feature] = count / total_depth_count

    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Depth', 'Feature', 'Probability'])
        for depth, probabilities in total_feature_probabilities.items():
            for feature, probability in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([depth, feature, probability])
    print(f"Results written to {output_csv_file}")
