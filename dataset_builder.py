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
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.avif', '.webp'))]
        
        for file in files:
            filepath = os.path.join(cat_dir, file)
            try:
                # 1. Original Image
                features = extract_features(filepath)
                dataset[cat].append([round(f, 4) for f in features])
                
                # 2. DATA AUGMENTATION (Multiply what the AI "sees")
                # We open the image and create virtual variations
                from PIL import Image, ImageEnhance
                with Image.open(filepath) as img:
                    img = img.convert("RGB")
                    
                    # Variation A: Horizontal Flip (Mirror)
                    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                    # We need a temp path or a modified extract_features that accepts Image objects
                    # For simplicity, let's create a temp file
                    temp_path = os.path.join(os.path.dirname(filepath), "_temp_aug.png")
                    
                    # Flip
                    flipped.save(temp_path)
                    dataset[cat].append([round(f, 4) for f in extract_features(temp_path)])
                    
                    # Variation B: Brighter
                    ImageEnhance.Brightness(img).enhance(1.2).save(temp_path)
                    dataset[cat].append([round(f, 4) for f in extract_features(temp_path)])
                    
                    # Variation C: Darker
                    ImageEnhance.Brightness(img).enhance(0.8).save(temp_path)
                    dataset[cat].append([round(f, 4) for f in extract_features(temp_path)])
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                total_images += 4 # 1 original + 3 variations
                print(f"  [OK] Successfully learned 4 variations for: {cat}/{file}")
            except Exception as e:
                print(f"  [ERROR] Processing {file}: {e}")
                
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f)
        
    print("=" * 40)
    print(f"Success! {total_images} feature vectors saved to {OUTPUT_FILE}")
    print("Your app will now load instantly on Render!")

if __name__ == "__main__":
    build_dataset()
