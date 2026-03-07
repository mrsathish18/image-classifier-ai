import os
import json

def load_dataset_features(json_path, classifier):
    """
    Instantly loads pre-computed vectors from JSON into the classifier.
    Avoids opening heavy image files on the Render server, saving >450MB of RAM.
    """
    if not os.path.exists(json_path):
        print(f"Feature dataset not found at {json_path}. Run dataset_builder.py first!")
        return 0
        
    with open(json_path, 'r') as f:
        dataset = json.load(f)
        
    count = 0
    for category, feature_lists in dataset.items():
        for features in feature_lists:
            classifier.add_training_data(category, features)
            count += 1
            
    return count
