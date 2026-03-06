"""
training_loader.py — Loads training images from folders

WHAT THIS FILE DOES:
    Scans the training_data/ folder structure and extracts features
    from all images, organized by category.

FOLDER STRUCTURE:
    training_data/
    ├── car/
    │   ├── car1.jpg
    │   ├── car2.jpg
    │   └── car3.jpg
    ├── bike/
    │   ├── bike1.jpg
    │   └── bike2.jpg
    └── cat/
        ├── cat1.png
        └── cat2.png

Each subfolder name becomes the LABEL for all images inside it.
"""

import os
from feature_extractor import extract_features
from knn_classifier import KNNClassifier


# Image file extensions we support
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def load_training_data(training_dir, classifier):
    """
    Loads all training images and adds them to the classifier.
    
    Args:
        training_dir: path to the training_data/ folder
        classifier: a KNNClassifier instance
    
    Returns:
        dict with stats: {"total": 25, "categories": {"car": 10, "bike": 8, ...}}
    """
    stats = {"total": 0, "categories": {}}
    
    # Check if training directory exists
    if not os.path.exists(training_dir):
        print(f"ERROR: Training directory '{training_dir}' not found!")
        print("Please create it and add image folders inside.")
        return stats
    
    # List all subfolders (each subfolder = one category)
    categories = [
        name for name in os.listdir(training_dir)
        if os.path.isdir(os.path.join(training_dir, name))
    ]
    
    if len(categories) == 0:
        print("ERROR: No category folders found in training_data/")
        print("Please create folders like: training_data/car/, training_data/bike/")
        return stats
    
    print(f"Found {len(categories)} categories: {', '.join(categories)}")
    print("-" * 50)
    
    # Process each category folder
    for category in sorted(categories):
        category_path = os.path.join(training_dir, category)
        image_count = 0
        
        # List all image files in this category folder
        for filename in os.listdir(category_path):
            # Check if it's an image file
            _, ext = os.path.splitext(filename)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue  # Skip non-image files
            
            filepath = os.path.join(category_path, filename)
            
            try:
                # Extract features from this image
                features = extract_features(filepath)
                
                # Add to the classifier with the category as the label
                classifier.add_training_data(category, features)
                
                image_count += 1
                stats["total"] += 1
                
            except Exception as e:
                print(f"  WARNING: Could not process '{filename}': {e}")
        
        stats["categories"][category] = image_count
        print(f"  [{category}] Loaded {image_count} images")
    
    print("-" * 50)
    print(f"Total: {stats['total']} training images loaded")
    
    return stats


def validate_training_data(training_dir):
    """
    Checks if the training data is set up correctly.
    Returns a list of issues found (empty list = all good).
    """
    issues = []
    
    if not os.path.exists(training_dir):
        issues.append(f"Training directory '{training_dir}' does not exist")
        return issues
    
    categories = [
        name for name in os.listdir(training_dir)
        if os.path.isdir(os.path.join(training_dir, name))
    ]
    
    if len(categories) == 0:
        issues.append("No category folders found (need at least 2)")
        return issues
    
    if len(categories) == 1:
        issues.append("Only 1 category found — need at least 2 for classification")
    
    for category in categories:
        category_path = os.path.join(training_dir, category)
        images = [
            f for f in os.listdir(category_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        
        if len(images) == 0:
            issues.append(f"'{category}/' folder is empty — add some images!")
        elif len(images) < 3:
            issues.append(f"'{category}/' has only {len(images)} images — add at least 5 for better accuracy")
    
    return issues
