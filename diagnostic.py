"""
diagnostic.py — Checks every module to find the "unhashable list" error
"""

import os
import sys

# Ensure current dir is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from feature_extractor import extract_features
    from knn_classifier import KNNClassifier
    from training_loader import load_training_data
    print("Step 1: Imports successful")
except Exception as e:
    print(f"Step 1 Failed: {e}")
    sys.exit(1)

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")

def run_diagnostic():
    print(f"Step 2: Checking training directory: {TRAINING_DIR}")
    if not os.path.exists(TRAINING_DIR):
        print("ERROR: Training directory missing")
        return

    classifier = KNNClassifier(k=3)
    
    print("Step 3: Loading training data...")
    try:
        stats = load_training_data(TRAINING_DIR, classifier)
        print(f"Stats: {stats}")
    except Exception as e:
        print(f"ERROR during loading: {e}")
        import traceback
        traceback.print_exc()
        return

    if classifier.get_training_count() == 0:
        print("ERROR: No images loaded. Please add some images.")
        return

    print("Step 4: Testing classification on a training image...")
    try:
        # Pick the first training image we find
        for cat in os.listdir(TRAINING_DIR):
            cat_path = os.path.join(TRAINING_DIR, cat)
            if os.path.isdir(cat_path):
                imgs = [f for f in os.listdir(cat_path) if f.endswith(('.png', '.jpg'))]
                if imgs:
                    test_path = os.path.join(cat_path, imgs[0])
                    print(f"Testing with: {test_path}")
                    
                    print("  Extracting features...")
                    features = extract_features(test_path)
                    print(f"  Features type: {type(features)}")
                    
                    print("  Classifying...")
                    result = classifier.classify_with_details(features)
                    print(f"  Result: {result['label']} ({result['confidence']})")
                    break
    except Exception as e:
        print(f"ERROR during classification: {e}")
        import traceback
        traceback.print_exc()

    print("\nDIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    run_diagnostic()
