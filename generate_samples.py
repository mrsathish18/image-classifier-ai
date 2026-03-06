"""
generate_samples.py — Creates sample training images for testing

RUN THIS FIRST to generate simple test images, so you can see the app work!

Command: python generate_samples.py
Then:    python app.py
"""

from PIL import Image, ImageDraw
import os
import random

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")

# Each category has a distinct color/pattern so KNN can tell them apart
CATEGORY_STYLES = {
    "car": {
        "bg_colors": [(180, 30, 30), (200, 50, 40), (160, 20, 20), (190, 40, 35), (170, 25, 25)],
        "shape": "rectangle_wide",  # Cars are wide rectangles
    },
    "bike": {
        "bg_colors": [(30, 30, 180), (40, 50, 200), (20, 20, 160), (35, 40, 190), (25, 25, 170)],
        "shape": "circles",  # Bikes have wheels
    },
    "cat": {
        "bg_colors": [(200, 150, 50), (180, 130, 40), (210, 160, 60), (190, 140, 45), (195, 145, 55)],
        "shape": "triangle",  # Cat ears
    },
    "dog": {
        "bg_colors": [(139, 90, 43), (120, 75, 35), (150, 100, 50), (130, 85, 40), (140, 95, 45)],
        "shape": "oval",  # Dog face
    },
    "flower": {
        "bg_colors": [(220, 50, 180), (200, 40, 160), (230, 60, 190), (210, 45, 170), (215, 55, 175)],
        "shape": "star",  # Flower petals
    },
}


def create_sample_image(bg_color, shape, size=(128, 128)):
    """Creates a simple colored image with a shape."""
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    cx, cy = size[0] // 2, size[1] // 2
    noise_x = random.randint(-10, 10)
    noise_y = random.randint(-10, 10)

    if shape == "rectangle_wide":
        draw.rectangle([20 + noise_x, 40 + noise_y, 108 + noise_x, 90 + noise_y],
                       fill=(min(bg_color[0]+60,255), min(bg_color[1]+60,255), min(bg_color[2]+60,255)))

    elif shape == "circles":
        r = 20
        draw.ellipse([25 + noise_x, 50 + noise_y, 25+2*r + noise_x, 50+2*r + noise_y],
                     fill=(min(bg_color[0]+80,255), min(bg_color[1]+80,255), min(bg_color[2]+80,255)))
        draw.ellipse([75 + noise_x, 50 + noise_y, 75+2*r + noise_x, 50+2*r + noise_y],
                     fill=(min(bg_color[0]+80,255), min(bg_color[1]+80,255), min(bg_color[2]+80,255)))

    elif shape == "triangle":
        points = [(cx + noise_x, 20 + noise_y), (30 + noise_x, 100 + noise_y), (98 + noise_x, 100 + noise_y)]
        draw.polygon(points, fill=(min(bg_color[0]+50,255), min(bg_color[1]+50,255), min(bg_color[2]+50,255)))

    elif shape == "oval":
        draw.ellipse([30 + noise_x, 25 + noise_y, 100 + noise_x, 105 + noise_y],
                     fill=(min(bg_color[0]+40,255), min(bg_color[1]+40,255), min(bg_color[2]+40,255)))

    elif shape == "star":
        for angle_offset in range(0, 360, 72):
            import math
            rad = math.radians(angle_offset + noise_x)
            px = int(cx + 35 * math.cos(rad))
            py = int(cy + 35 * math.sin(rad))
            draw.ellipse([px-12, py-12, px+12, py+12],
                         fill=(min(bg_color[0]+70,255), min(bg_color[1]+70,255), min(bg_color[2]+70,255)))

    return img


def generate_all():
    """Generates 8 sample images per category."""
    print("Generating sample training images...")
    print("=" * 40)

    for category, style in CATEGORY_STYLES.items():
        folder = os.path.join(TRAINING_DIR, category)
        os.makedirs(folder, exist_ok=True)

        for i in range(8):
            bg = style["bg_colors"][i % len(style["bg_colors"])]
            # Add slight random variation
            bg = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in bg)
            img = create_sample_image(bg, style["shape"])
            filepath = os.path.join(folder, f"{category}_{i+1}.png")
            img.save(filepath)

        print(f"  [OK] {category}/ -> 8 images created")

    print("=" * 40)
    print("Done! Now run: python app.py")


if __name__ == "__main__":
    generate_all()
