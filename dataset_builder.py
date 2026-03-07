import os
import json
from feature_extractor import extract_features

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_features.json')

def build_dataset():
    if not os.path.exists(TRAINING_DIR):
        print(f"Training directory not found at {TRAINING_DIR}!")
        return
        
    dataset = {}
    categories = [d for d in os.listdir(TRAINING_DIR) if os.path.isdir(os.path.join(TRAINING_DIR, d))]
    
    total_images = 0
    print("Starting Feature Extraction (300+ features per image)...")
    
    for cat in categories:
        dataset[cat] = []
        cat_dir = os.path.join(TRAINING_DIR, cat)
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for file in files:
            filepath = os.path.join(cat_dir, file)
            try:
                features = extract_features(filepath)
                # Save just the numbers to memory, saving MASSIVE RAM
                dataset[cat].append([round(f, 4) for f in features])
                total_images += 1
                print(f"  [OK] Calculated {len(features)} features for: {cat}/{file}")
            except Exception as e:
                print(f"  [ERROR] Processing {file}: {e}")
                
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f)
        
    print("=" * 40)
    print(f"Success! {total_images} feature vectors saved to {OUTPUT_FILE}")
    print("Your app will now load instantly on Render!")

if __name__ == "__main__":
    build_dataset()
