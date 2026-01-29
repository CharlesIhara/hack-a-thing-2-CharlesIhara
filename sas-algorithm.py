import cv2
import numpy as np

def compute_edges_and_orientation(img_gray):
    # Sobel gradients
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)  # range: [-π, π]

    return magnitude, orientation

def quantize_orientation(orientation, R):
    # Map [-π, π] → [0, R)
    orientation = (orientation + np.pi) * R / (2 * np.pi)
    bins = np.floor(orientation).astype(int)
    bins = np.clip(bins, 0, R - 1)
    return bins

def build_orientation_maps(magnitude, orientation_bins, R, mag_thresh=10):
    h, w = magnitude.shape
    maps = np.zeros((R, h, w), dtype=np.float32)

    edge_mask = magnitude > mag_thresh

    for r in range(R):
        mask = (orientation_bins == r) & edge_mask
        maps[r][mask] = magnitude[mask]

    return maps

def spatial_reduce(orientation_maps, N):
    R, H, W = orientation_maps.shape
    reduced = np.zeros((R, N, N), dtype=np.float32)

    cell_h = H // N
    cell_w = W // N

    for r in range(R):
        for i in range(N):
            for j in range(N):
                y0 = i * cell_h
                y1 = (i + 1) * cell_h
                x0 = j * cell_w
                x1 = (j + 1) * cell_w
                reduced[r, i, j] = orientation_maps[r, y0:y1, x0:x1].mean()

    return reduced

def create_template(patterns):
    # patterns: list of arrays with shape (R, N, N)
    return np.mean(patterns, axis=0)


def similarity(template, sample):
    t = template.flatten()
    s = sample.flatten()

    numerator = np.dot(t, s)
    denominator = np.linalg.norm(t) * np.linalg.norm(s)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def authenticate(template, sample, threshold=0.8):
    score = similarity(template, sample)
    return score >= threshold, score

def extract_edge_orientation_pattern(img_gray, R=4, N=8):
    magnitude, orientation = compute_edges_and_orientation(img_gray)
    bins = quantize_orientation(orientation, R)
    maps = build_orientation_maps(magnitude, bins, R)
    pattern = spatial_reduce(maps, N)
    return pattern



import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load sketch
img = cv2.imread("signature.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Invert if needed (white strokes on black)
if img.mean() > 127:
    img = 255 - img

plt.imshow(img, cmap="gray")
plt.title("Input Sketch")
plt.axis("off")