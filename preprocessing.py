import math

def apply_gaussian_blur(gray_pixels, width, height):
    """Applies a 3x3 Gaussian Blur to reduce image noise."""
    kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
    kernel_sum = 16
    
    blurred = list(gray_pixels)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            val = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    val += gray_pixels[(y + ky) * width + (x + kx)] * kernel[ky + 1][kx + 1]
            blurred[y * width + x] = val // kernel_sum
    return blurred

def grayscale(pixels):
    """Converts RGB tuples to standard grayscale."""
    return [int(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in pixels]

def normalize_features(features):
    """Min-Max normalization to keep vector values between 0.0 and 1.0."""
    if not features: return features
    min_val = min(features)
    max_val = max(features)
    if max_val - min_val == 0:
        return [0.0] * len(features)
    return [(f - min_val) / (max_val - min_val) for f in features]
