# 🔍 Image Classifier — Pure Python (No Third-Party ML)

A KNN-based image classifier. Upload any image → it tells you what it is!

**Built without any third-party ML libraries** — no TensorFlow, no OpenCV, no sklearn.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Pillow (for reading images)
```bash
pip install Pillow
```

### Step 2: Add Training Images
Put 5-10 images in each category folder:
```
training_data/
├── car/         ← put 5-10 car images (.jpg or .png)
├── bike/        ← put 5-10 bike images
├── cat/         ← put 5-10 cat images
├── dog/         ← put 5-10 dog images
└── flower/      ← put 5-10 flower images
```

> **TIP:** Download images from Google Images. More images = better accuracy!

### Step 3: Run the App
```bash
python desktop_app.py
```

Click **"Choose Image"** → Select any image → See the result!

---

## 📁 Project Files Explained

| File | What It Does |
|------|-------------|
| `desktop_app.py` | **Main file** — Run this! Opens the GUI window |
| `feature_extractor.py` | Extracts color/edge features from images |
| `knn_classifier.py` | KNN algorithm (the "brain") — from scratch! |
| `training_loader.py` | Loads training images from folders |
| `training_data/` | Folder with category subfolders of images |

---

## ➕ Add New Categories

Want to detect "airplane" or "phone"? Easy:

1. Create new folder: `training_data/airplane/`
2. Add 5-10 airplane images inside
3. Click **"🔄 Reload Training Data"** in the app
4. Done! It can now detect airplanes.

---

## 🧠 How It Works (Simple)

```
Image → Extract Features (colors, edges) → Compare to Training → Closest Match = Answer
```

1. **Feature Extraction:** Each image is converted to 69 numbers describing its colors, edges, and shape
2. **KNN Algorithm:** When you upload a new image, it finds the 5 most similar training images
3. **Majority Vote:** If 3 out of 5 closest matches are "car" → Output is "car"!

---

## ⚠️ Accuracy Tips

- Add **at least 5 images per category** (10+ is better)
- Use **varied images** (different angles, lighting, backgrounds)
- Works best with **5-10 categories** (too many reduces accuracy)
- Expected accuracy: **60-75%** for few categories with good training data
