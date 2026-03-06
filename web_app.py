import os
import time
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from feature_extractor import extract_features
from knn_classifier import KNNClassifier
from training_loader import load_training_data

# Use absolute paths for stability on deployment servers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
TRAINING_DIR = os.path.join(BASE_DIR, 'training_data')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
K_NEIGHBORS = 5

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)

# Debug: Print directory structure
print(f"--- DEBUG: Files in {BASE_DIR} ---")
try:
    for root, dirs, files in os.walk(BASE_DIR):
        level = root.replace(BASE_DIR, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        # Limit the number of files shown to keep logs clean
        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... ({len(files)-5} more files)")
        if level >= 2: break # Don't go too deep
except Exception as e:
    print(f"Debug error: {e}")
print("---------------------------------")

# Initialize Classifier and load data once at startup
classifier = KNNClassifier(k=K_NEIGHBORS)
print("Loading training data for web app...")
load_training_data(TRAINING_DIR, classifier)
print(f"Loaded {classifier.get_training_count()} images.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    categories = classifier.get_categories()
    return render_template('index.html', categories=categories, count=classifier.get_training_count())

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file segment'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Step 1: Extract features
            features = extract_features(filepath)
            
            # Step 2: Classify
            result = classifier.classify_with_details(features)
            
            # Step 3: Format response
            response = {
                'label': result['label'],
                'confidence': round(result['confidence'] * 100, 1),
                'all_scores': {k: round(v * 100, 1) for k, v in result['all_scores'].items()},
                'image_url': url_for('static', filename=f'uploads/{filename}')
            }
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f"Processing error: {str(e)}"}), 500
            
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    # For local running
    app.run(debug=True, port=5000)
