"""
app.py — Main Application with GUI (Graphical User Interface)

THIS IS THE FILE YOU RUN!
    Command: python app.py

WHAT IT DOES:
    1. Opens a window with a "Choose Image" button
    2. You click the button → file manager opens
    3. You select an image (car.jpg, bike.png, etc.)
    4. The app shows the image and tells you: "This is a CAR! (75% confidence)"

USES:
    - tkinter (built-in Python GUI library — NOT third-party!)
    - Pillow (PIL) — only for reading/displaying images
    - Our custom modules for the actual classification
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

from feature_extractor import extract_features
from knn_classifier import KNNClassifier
from training_loader import load_training_data, validate_training_data

# Try to import PIL for image display in the GUI
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ========================
# CONFIGURATION
# ========================
TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")
K_NEIGHBORS = 5  # Number of neighbors to check (odd number is best)


# ========================
# COLOR THEME (Dark Mode)
# ========================
COLORS = {
    'bg_dark':      '#1a1a2e',      # Main background
    'bg_card':      '#16213e',      # Card background
    'bg_input':     '#0f3460',      # Input area background
    'accent':       '#e94560',      # Accent color (red-pink)
    'accent_hover': '#ff6b6b',      # Accent hover
    'success':      '#00d2d3',      # Success/confidence color
    'text_white':   '#ffffff',      # Primary text
    'text_gray':    '#a0a0b0',      # Secondary text
    'text_dim':     '#606080',      # Dimmed text
    'border':       '#2a2a4a',      # Borders
}


class ImageClassifierApp:
    """The main application window."""
    
    def __init__(self):
        # ========== Create the main window ==========
        self.root = tk.Tk()
        self.root.title("🔍 Image Classifier — KNN (No Third-Party ML)")
        self.root.geometry("900x700")
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.resizable(True, True)
        
        # ========== Initialize the classifier ==========
        self.classifier = KNNClassifier(k=K_NEIGHBORS)
        self.current_image_path = None
        
        # ========== Build the GUI ==========
        self._build_gui()
        
        # ========== Load training data ==========
        self._load_training()
    
    def _build_gui(self):
        """Creates all the GUI elements (buttons, labels, etc.)."""
        
        # ---------- TITLE BAR ----------
        title_frame = tk.Frame(self.root, bg=COLORS['bg_dark'], pady=15)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="🔍 IMAGE CLASSIFIER",
            font=("Segoe UI", 22, "bold"),
            fg=COLORS['text_white'],
            bg=COLORS['bg_dark']
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="KNN-Based Classification • No Third-Party ML Libraries",
            font=("Segoe UI", 10),
            fg=COLORS['text_gray'],
            bg=COLORS['bg_dark']
        )
        subtitle_label.pack()
        
        # ---------- TRAINING STATUS BAR ----------
        self.status_frame = tk.Frame(self.root, bg=COLORS['bg_card'], pady=8, padx=15)
        self.status_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        self.training_status_label = tk.Label(
            self.status_frame,
            text="⏳ Loading training data...",
            font=("Segoe UI", 10),
            fg=COLORS['text_gray'],
            bg=COLORS['bg_card'],
            anchor="w"
        )
        self.training_status_label.pack(fill=tk.X)
        
        # ---------- MAIN CONTENT AREA ----------
        content_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # Left side: Image Preview
        left_frame = tk.Frame(content_frame, bg=COLORS['bg_card'], padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        img_title = tk.Label(
            left_frame,
            text="📷 IMAGE PREVIEW",
            font=("Segoe UI", 11, "bold"),
            fg=COLORS['text_gray'],
            bg=COLORS['bg_card']
        )
        img_title.pack(pady=(0, 10))
        
        # Image display area
        self.image_label = tk.Label(
            left_frame,
            text="No image selected\n\nClick the button below\nto choose an image",
            font=("Segoe UI", 12),
            fg=COLORS['text_dim'],
            bg=COLORS['bg_input'],
            width=40,
            height=15,
            relief=tk.FLAT
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # File name display
        self.filename_label = tk.Label(
            left_frame,
            text="",
            font=("Segoe UI", 9),
            fg=COLORS['text_dim'],
            bg=COLORS['bg_card']
        )
        self.filename_label.pack()
        
        # Right side: Results
        right_frame = tk.Frame(content_frame, bg=COLORS['bg_card'], padx=15, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        result_title = tk.Label(
            right_frame,
            text="🎯 CLASSIFICATION RESULT",
            font=("Segoe UI", 11, "bold"),
            fg=COLORS['text_gray'],
            bg=COLORS['bg_card']
        )
        result_title.pack(pady=(0, 15))
        
        # Main result label (big text showing "CAR", "BIKE", etc.)
        self.result_label = tk.Label(
            right_frame,
            text="—",
            font=("Segoe UI", 36, "bold"),
            fg=COLORS['accent'],
            bg=COLORS['bg_card']
        )
        self.result_label.pack(pady=(10, 5))
        
        # Confidence label
        self.confidence_label = tk.Label(
            right_frame,
            text="",
            font=("Segoe UI", 14),
            fg=COLORS['success'],
            bg=COLORS['bg_card']
        )
        self.confidence_label.pack(pady=(0, 20))
        
        # Detailed scores frame
        self.scores_frame = tk.Frame(right_frame, bg=COLORS['bg_card'])
        self.scores_frame.pack(fill=tk.X, pady=(0, 10))
        
        scores_title = tk.Label(
            self.scores_frame,
            text="Category Scores:",
            font=("Segoe UI", 10, "bold"),
            fg=COLORS['text_gray'],
            bg=COLORS['bg_card'],
            anchor="w"
        )
        scores_title.pack(fill=tk.X, pady=(0, 5))
        
        self.scores_container = tk.Frame(self.scores_frame, bg=COLORS['bg_card'])
        self.scores_container.pack(fill=tk.X)
        
        # ---------- BUTTON BAR ----------
        button_frame = tk.Frame(self.root, bg=COLORS['bg_dark'], pady=15)
        button_frame.pack(fill=tk.X, padx=20)
        
        # Choose Image button
        self.choose_btn = tk.Button(
            button_frame,
            text="📁  CHOOSE IMAGE FROM FILE MANAGER",
            font=("Segoe UI", 13, "bold"),
            fg=COLORS['text_white'],
            bg=COLORS['accent'],
            activebackground=COLORS['accent_hover'],
            activeforeground=COLORS['text_white'],
            relief=tk.FLAT,
            cursor="hand2",
            padx=30,
            pady=12,
            command=self._choose_image
        )
        self.choose_btn.pack(fill=tk.X)
        
        # Retrain button
        retrain_btn = tk.Button(
            button_frame,
            text="🔄 Reload Training Data",
            font=("Segoe UI", 9),
            fg=COLORS['text_gray'],
            bg=COLORS['bg_card'],
            activebackground=COLORS['border'],
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=5,
            command=self._load_training
        )
        retrain_btn.pack(fill=tk.X, pady=(8, 0))
    
    def _load_training(self):
        """Loads training images from the training_data/ folder."""
        self.training_status_label.config(text="⏳ Loading training data...")
        self.root.update()
        
        # Validate first
        issues = validate_training_data(TRAINING_DIR)
        
        if issues:
            warning_text = "⚠️ Training Data Issues:\n"
            for issue in issues:
                warning_text += f"  • {issue}\n"
            
            self.training_status_label.config(
                text=f"⚠️ Issues found — {len(issues)} warning(s). Check console.",
                fg="#ffaa00"
            )
            print("\n" + "=" * 50)
            print(warning_text)
            print("=" * 50)
            
            # Still try to load what we can
            if not os.path.exists(TRAINING_DIR):
                self._create_sample_folders()
                messagebox.showinfo(
                    "Training Data Needed",
                    f"Created folder: {TRAINING_DIR}\n\n"
                    "Please add image folders inside:\n"
                    "  training_data/car/   ← put car images here\n"
                    "  training_data/bike/  ← put bike images here\n"
                    "  training_data/cat/   ← put cat images here\n\n"
                    "Then click '🔄 Reload Training Data'"
                )
                return
        
        # Reset classifier and reload
        self.classifier = KNNClassifier(k=K_NEIGHBORS)
        stats = load_training_data(TRAINING_DIR, self.classifier)
        
        if stats["total"] > 0:
            categories = ", ".join(
                f"{cat}({count})" 
                for cat, count in stats["categories"].items()
            )
            self.training_status_label.config(
                text=f"✅ Loaded {stats['total']} images | Categories: {categories}",
                fg=COLORS['success']
            )
        else:
            self.training_status_label.config(
                text="❌ No training images found. Add images to training_data/ folders.",
                fg=COLORS['accent']
            )
    
    def _create_sample_folders(self):
        """Creates the training_data/ folder with sample category subfolders."""
        categories = ["car", "bike", "cat", "dog", "flower"]
        for cat in categories:
            folder = os.path.join(TRAINING_DIR, cat)
            os.makedirs(folder, exist_ok=True)
        print(f"Created training folders at: {TRAINING_DIR}")
    
    def _choose_image(self):
        """Opens file manager dialog to select an image."""
        filepath = filedialog.askopenfilename(
            title="Select an Image to Classify",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All Files", "*.*")
            ]
        )
        
        if not filepath:
            return  # User cancelled
        
        self.current_image_path = filepath
        
        # Show the selected image in the preview
        self._show_image_preview(filepath)
        
        # Show filename
        self.filename_label.config(text=os.path.basename(filepath))
        
        # Check if we have training data
        if self.classifier.get_training_count() == 0:
            self.result_label.config(text="NO DATA", fg="#ffaa00")
            self.confidence_label.config(text="Add training images first!")
            messagebox.showwarning(
                "No Training Data",
                "Please add images to the training_data/ folders first!\n\n"
                "Example:\n"
                "  training_data/car/ ← put 5-10 car images\n"
                "  training_data/bike/ ← put 5-10 bike images\n\n"
                "Then click '🔄 Reload Training Data'"
            )
            return
        
        # Classify the image!
        self._classify_image(filepath)
    
    def _show_image_preview(self, filepath):
        """Displays the selected image in the preview area."""
        if not PIL_AVAILABLE:
            self.image_label.config(text=f"[Image: {os.path.basename(filepath)}]")
            return
        
        try:
            img = Image.open(filepath)
            
            # Resize to fit the preview area (max 350x300)
            img.thumbnail((350, 300))
            
            # Convert for tkinter display
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference to prevent garbage collection
            
        except Exception as e:
            self.image_label.config(text=f"Could not display image:\n{e}")
    
    def _classify_image(self, filepath):
        """Extracts features and classifies the image."""
        self.result_label.config(text="🔄", fg=COLORS['text_gray'])
        self.confidence_label.config(text="Analyzing...")
        self.root.update()
        
        try:
            # Step 1: Extract features from the new image
            features = extract_features(filepath)
            
            # Step 2: Classify using KNN
            result = self.classifier.classify_with_details(features)
            
            # Step 3: Display the result!
            label = result['label'].upper()
            confidence = result['confidence']
            
            # Color based on confidence
            if confidence >= 0.7:
                color = "#00d2d3"  # Teal = high confidence
            elif confidence >= 0.4:
                color = "#ffaa00"  # Orange = medium confidence
            else:
                color = "#ff6b6b"  # Red = low confidence
            
            self.result_label.config(text=label, fg=color)
            self.confidence_label.config(
                text=f"Confidence: {confidence:.0%}",
                fg=color
            )
            
            # Show all category scores
            self._show_scores(result['all_scores'])
            
        except Exception as e:
            self.result_label.config(text="ERROR", fg=COLORS['accent'])
            self.confidence_label.config(text=str(e))
    
    def _show_scores(self, all_scores):
        """Displays score bars for each category."""
        # Clear previous scores
        for widget in self.scores_container.winfo_children():
            widget.destroy()
        
        # Sort by score (highest first)
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        for category, score in sorted_scores:
            row = tk.Frame(self.scores_container, bg=COLORS['bg_card'])
            row.pack(fill=tk.X, pady=2)
            
            # Category name
            name_label = tk.Label(
                row,
                text=f"{category}:",
                font=("Segoe UI", 10),
                fg=COLORS['text_gray'],
                bg=COLORS['bg_card'],
                width=12,
                anchor="w"
            )
            name_label.pack(side=tk.LEFT)
            
            # Score bar background
            bar_bg = tk.Frame(row, bg=COLORS['bg_input'], height=18)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
            bar_bg.pack_propagate(False)
            
            # Score bar fill
            bar_width = max(int(score * 100), 1)
            bar_color = COLORS['success'] if score >= 0.5 else COLORS['text_dim']
            bar_fill = tk.Frame(bar_bg, bg=bar_color, width=bar_width)
            bar_fill.pack(side=tk.LEFT, fill=tk.Y)
            
            # Score percentage
            pct_label = tk.Label(
                row,
                text=f"{score:.0%}",
                font=("Segoe UI", 10, "bold"),
                fg=COLORS['text_white'] if score >= 0.5 else COLORS['text_dim'],
                bg=COLORS['bg_card'],
                width=5,
                anchor="e"
            )
            pct_label.pack(side=tk.RIGHT)
    
    def run(self):
        """Starts the application."""
        print("\nImage Classifier Started!")
        print(f"   Training data folder: {TRAINING_DIR}")
        print(f"   K-Neighbors: {K_NEIGHBORS}")
        print("   Close the window to exit.\n")
        self.root.mainloop()


# ========================
# RUN THE APP
# ========================
if __name__ == "__main__":
    app = ImageClassifierApp()
    app.run()
