import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load sketch
img = cv2.imread("signature.png", cv2.IMREAD_GRAYSCALE)

# resize
def resize_with_padding(img, target=256):
    h, w = img.shape
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((target, target), dtype=np.uint8)
    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2

    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas
img = resize_with_padding(img)

# Invert if needed (white strokes on black)
if img.mean() > 127:
    img = 255 - img

# fix gradients
_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

plt.imshow(img, cmap="gray")
plt.title("Input Sketch")
plt.axis("off")

def extract_edge_orientation_pattern(img_gray, R=4, N=8):
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    gx = cv2.Sobel(img_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_blur, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)

    # Quantize orientation
    orientation = (orientation + np.pi) * R / (2 * np.pi)
    bins = np.floor(orientation).astype(int)
    bins = np.clip(bins, 0, R - 1)

    # Build orientation maps
    maps = np.zeros((R, img_gray.shape[0], img_gray.shape[1]), dtype=np.float32)
    edge_mask = magnitude > 10

    for r in range(R):
        maps[r][(bins == r) & edge_mask] = magnitude[(bins == r) & edge_mask]

    # Spatial reduction
    H, W = img_gray.shape
    reduced = np.zeros((R, N, N), dtype=np.float32)
    cell_h, cell_w = H // N, W // N

    for r in range(R):
        for i in range(N):
            for j in range(N):
                reduced[r, i, j] = maps[r,
                                        i*cell_h:(i+1)*cell_h,
                                        j*cell_w:(j+1)*cell_w].mean()

    return reduced


R = 4
N = 8

pattern = extract_edge_orientation_pattern(img, R, N)
template = pattern.copy()  # one-sample enrollment

def similarity(a, b, eps=1e-8):
    a = a.flatten()
    b = b.flatten()

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < eps:
        return 0.0

    return np.dot(a, b) / denom

score = similarity(template, pattern)
print("Similarity:", score)

THRESHOLD = 0.85
accepted = score >= THRESHOLD

print("Accepted:", accepted)