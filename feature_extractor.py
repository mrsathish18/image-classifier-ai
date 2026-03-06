"""
feature_extractor.py — Extracts a "fingerprint" from any image

WHAT THIS FILE DOES:
    Takes an image and converts it into numbers (a "feature vector").
    These numbers describe the image's colors, edges, and shape.
    
WHY WE NEED THIS:
    Computers can't "see" images like humans. They need numbers.
    We convert each image into ~69 numbers that describe it.
    Then we can compare these numbers to classify images.

NO THIRD-PARTY ML LIBRARIES — only Pillow for reading the image file.
"""

from PIL import Image  # ONLY used to read image files — NOT for detection
import math


def extract_features(image_path):
    """
    Takes an image file path → Returns a list of numbers (features).
    
    Example:
        features = extract_features("car.jpg")
        # features = [0.12, 0.05, 0.34, ...] (69 numbers)
    """
    # Step 1: Open the image and resize to 64x64
    # WHY RESIZE? So all images are the same size for fair comparison
    with Image.open(image_path) as img:
        # Use thumbnail first to drastically reduce memory usage for large 4K images
        img.thumbnail((256, 256)) 
        image = img.convert("RGB")  
        image = image.resize((64, 64))
    
    # Step 2: Get all pixels as a list of RGB tuples
    # We must ensure every pixel is a tuple (not a list) to avoid "unhashable" errors
    # We also handle grayscale (p is int) and RGBA (p is 4-tuple)
    pixels = []
    for p in image.getdata():
        if isinstance(p, (list, tuple)):
            # If it's a list/tuple, take first 3 values (R, G, B) and ensure they are ints
            pixels.append((int(p[0]), int(p[1]), int(p[2])))
        elif isinstance(p, int):
            # If it's an integer (grayscale), convert to RGB tuple
            pixels.append((p, p, p))
        else:
            # Fallback for any other type, try to convert to list then tuple
            try:
                p_list = list(p)
                pixels.append((int(p_list[0]), int(p_list[1]), int(p_list[2])))
            except:
                pixels.append((0, 0, 0)) # Last resort: black pixel
    
    width, height = image.size
    
    # Step 3: Extract different types of features
    color_hist = _color_histogram(pixels)        # 64 numbers (0.0 to 1.0)
    edge_density = _edge_density(pixels, width, height)  # 1 number (0.0 to 1.0)
    dominant_color = _dominant_color(pixels)      # 3 numbers (0.0 to 1.0)
    aspect_ratio = _aspect_ratio(image_path)      # 1 number (usually 0.5 to 2.0)
    
    # Step 4: Apply Weights!
    # WHY WEIGHTS? If we don't weight them, a small change in Aspect Ratio
    # might overpower a massive difference in Color.
    # We want Color and Edges to be the most important things for the AI.
    
    # Weight color heavily (x 2.0)
    weighted_color_hist = [c * 2.0 for c in color_hist]
    
    # Weight dominant color very heavily (x 3.0) because cars/flowers are color-coded
    weighted_dominant = [d * 3.0 for d in dominant_color]
    
    # Weight edges moderately (x 1.5) to separate smooth (flowers) from sharp (bikes)
    weighted_edge = [edge_density * 1.5]
    
    # Keep aspect ratio weak (x 0.5) because a side-profile bike and a portrait flower 
    # might accidentally have the same shape ratio.
    weighted_aspect = [aspect_ratio * 0.5]
    
    # Step 5: Combine all weighted features into one list
    features = weighted_color_hist + weighted_edge + weighted_dominant + weighted_aspect
    
    return features


def _color_histogram(pixels):
    """
    Creates a COLOR HISTOGRAM — counts how many pixels have each color.
    
    Think of it like sorting M&Ms by color and counting each pile.
    
    We divide each color channel (R, G, B) into 4 groups (called "bins"):
        Bin 0: values 0-63    (very dark)
        Bin 1: values 64-127  (dark)
        Bin 2: values 128-191 (light)
        Bin 3: values 192-255 (very light)
    
    4 bins × 4 bins × 4 bins = 64 total bins
    """
    # Create 64 bins, all starting at 0
    histogram = [0] * 64
    
    for r, g, b in pixels:
        # Divide each color by 64 to get which bin (0, 1, 2, or 3)
        r_bin = min(r // 64, 3)  # min() to handle edge case of value 256
        g_bin = min(g // 64, 3)
        b_bin = min(b // 64, 3)
        
        # Convert 3D bin position to 1D index
        # Example: r_bin=2, g_bin=1, b_bin=3 → index = 2*16 + 1*4 + 3 = 39
        index = r_bin * 16 + g_bin * 4 + b_bin
        histogram[index] += 1
    
    # Normalize: divide each count by total pixels
    # WHY? So the histogram works the same for any image size
    total = len(pixels)
    histogram = [count / total for count in histogram]
    
    return histogram  # Returns 64 numbers, each between 0.0 and 1.0


def _edge_density(pixels, width, height):
    """
    Calculates EDGE DENSITY — how many edges (boundaries) are in the image.
    
    Uses the SOBEL FILTER (a real computer vision algorithm).
    
    An "edge" is where the color changes sharply — like the boundary
    between a car and the road behind it.
    
    A landscape photo has FEW edges (smooth sky, smooth water).
    A car photo has MANY edges (sharp lines, wheels, windows).
    """
    # First convert to grayscale (single number per pixel instead of RGB)
    gray = []
    for r, g, b in pixels:
        # Standard grayscale formula (matches how human eyes work)
        gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
        gray.append(gray_value)
    
    # Sobel kernels — these are "templates" that detect edges
    # Horizontal edges detector
    sobel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    # Vertical edges detector
    sobel_y = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]
    
    edge_count = 0
    threshold = 100  # Only count strong edges
    
    # Scan every pixel (except borders — we need 3x3 area around each pixel)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = 0  # Horizontal edge strength
            gy = 0  # Vertical edge strength
            
            # Apply the 3x3 kernel around this pixel
            for ky in range(-1, 2):      # -1, 0, 1
                for kx in range(-1, 2):  # -1, 0, 1
                    # Get the gray value of the neighbor pixel
                    pixel_value = gray[(y + ky) * width + (x + kx)]
                    
                    # Multiply by the kernel weight
                    gx += pixel_value * sobel_x[ky + 1][kx + 1]
                    gy += pixel_value * sobel_y[ky + 1][kx + 1]
            
            # Combine horizontal and vertical edges
            magnitude = math.sqrt(gx * gx + gy * gy)
            
            if magnitude > threshold:
                edge_count += 1
    
    # Return edge density as a ratio (0.0 = no edges, 1.0 = all edges)
    total_pixels = (width - 2) * (height - 2)
    return edge_count / total_pixels if total_pixels > 0 else 0


def _dominant_color(pixels):
    """
    Finds the DOMINANT COLOR — the most common color in the image.
    
    We simplify colors by rounding them to the nearest 32.
    Example: (137, 200, 45) → (128, 192, 32)
    
    Then we count which simplified color appears most often.
    
    Cars might have dominant gray/silver.
    Forests might have dominant green.
    """
    # Count simplified color occurrences
    color_counts = {}
    
    for r, g, b in pixels:
        # Round to nearest 32 (simplifies 16 million colors to ~500)
        simple_r = (r // 32) * 32
        simple_g = (g // 32) * 32
        simple_b = (b // 32) * 32
        
        key = (simple_r, simple_g, simple_b)
        color_counts[key] = color_counts.get(key, 0) + 1
    
    # Find the most common color
    dominant = max(color_counts, key=color_counts.get)
    
    # Normalize to 0.0-1.0 range
    return [dominant[0] / 255.0, dominant[1] / 255.0, dominant[2] / 255.0]


def _aspect_ratio(image_path):
    """
    Calculates ASPECT RATIO — width divided by height.
    
    A wide landscape photo has ratio > 1.0
    A tall portrait photo has ratio < 1.0
    A square photo has ratio = 1.0
    
    Cars are usually in landscape photos (wide).
    People standing are usually in portrait photos (tall).
    """
    with Image.open(image_path) as image:
        width, height = image.size

    return width / height if height > 0 else 1.0
