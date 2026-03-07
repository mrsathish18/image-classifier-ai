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

from PIL import Image
import math
from preprocessing import apply_gaussian_blur, grayscale, normalize_features

def extract_features(image_path):
    with Image.open(image_path) as img:
        img.thumbnail((256, 256)) 
        image = img.convert("RGB")  
        image = image.resize((64, 64))
        
    pixels = list(image.getdata())
    width, height = image.size
    
    # Preprocessing
    gray_pixels = grayscale(pixels)
    blurred_gray = apply_gaussian_blur(gray_pixels, width, height)
    
    # 1. Advanced Color Histogram | 5x5x5 grouping = 125 features
    hist_125 = _advanced_color_histogram(pixels)
    # 2. Local Binary Patterns (Texture) | 64 features
    lbp_64 = _lbp_texture(blurred_gray, width, height)
    # 3. Zone Edge Density | 4x4 Grid = 16 features
    zone_edge_16 = _zone_edge_density(blurred_gray, width, height, grid=4)
    # 4. Gradient Orientations | 3x3 Grid x 8 angles = 72 features
    grad_72 = _gradient_orientation(blurred_gray, width, height, grid=3)
    # 5. Global Texture Contrast/Variance | 3 features
    stats_3 = _texture_stats(blurred_gray)
    # 6. Aspect Ratio | 1 feature
    aspect_1 = [width / height if height > 0 else 1.0]

    # Combine all logic (125 + 64 + 16 + 72 + 3 + 1 = 281 features)
    features = hist_125 + lbp_64 + zone_edge_16 + grad_72 + stats_3 + aspect_1
    return normalize_features(features)

def _advanced_color_histogram(pixels):
    bins = [0] * 125
    for r, g, b in pixels:
        bins[min(r // 52, 4) * 25 + min(g // 52, 4) * 5 + min(b // 52, 4)] += 1
    total = len(pixels)
    return [b/total for b in bins]

def _lbp_texture(gray, w, h):
    bins = [0] * 64
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = gray[y * w + x]
            code = 0
            if gray[(y-1)*w + (x-1)] > center: code |= 128
            if gray[(y-1)*w + x] > center: code |= 64
            if gray[(y-1)*w + (x+1)] > center: code |= 32
            if gray[y*w + (x+1)] > center: code |= 16
            if gray[(y+1)*w + (x+1)] > center: code |= 8
            if gray[(y+1)*w + x] > center: code |= 4
            if gray[(y+1)*w + (x-1)] > center: code |= 2
            if gray[y*w + (x-1)] > center: code |= 1
            bins[code // 4] += 1
    return [b/sum(bins) if sum(bins) > 1 else 0 for b in bins]

def _zone_edge_density(gray, w, h, grid=4):
    zone_edges = [0] * (grid * grid)
    zw, zh = w // grid, h // grid
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = gray[y*w + (x+1)] - gray[y*w + (x-1)]
            gy = gray[(y+1)*w + x] - gray[(y-1)*w + x]
            if abs(gx) + abs(gy) > 60:
                zone_edges[min(y // zh, grid-1) * grid + min(x // zw, grid-1)] += 1
    return [e / (zw * zh) for e in zone_edges]

def _gradient_orientation(gray, w, h, grid=3):
    bins = [0] * (grid * grid * 8)
    zw, zh = w // grid, h // grid
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = gray[y*w + (x+1)] - gray[y*w + (x-1)]
            gy = gray[(y+1)*w + x] - gray[(y-1)*w + x]
            if gx == 0 and gy == 0: continue
            direction = int((math.atan2(gy, gx) + math.pi) / (2 * math.pi) * 8) % 8
            bins[(min(y // zh, grid-1) * grid + min(x // zw, grid-1)) * 8 + direction] += 1
    total_bins = sum(bins)
    return [b/total_bins if total_bins > 0 else 0 for b in bins]

def _texture_stats(gray):
    mean = sum(gray) / len(gray)
    variance = sum((p - mean)**2 for p in gray) / len(gray)
    contrast = max(gray) - min(gray)
    return [mean/255.0, min(1.0, variance/(255*255)), contrast/255.0]
