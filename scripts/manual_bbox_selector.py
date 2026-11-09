
import cv2
from pathlib import Path

image_path = Path(
    "data/raw/mc_ocr_2021/train/images/"
    "mcocr_public_145013zykbw.jpg"
)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

# Resize for comfortable viewing while preserving aspect ratio
height, width = image.shape[:2]
scale = min(1.0, 900 / height, 1100 / width)

display = cv2.resize(
    image,
    (round(width * scale), round(height * scale))
)

# Drag a rectangle around the text.
# Press Enter or Space to confirm.
x, y, w, h = cv2.selectROI(
    "Select text bounding box",
    display,
    fromCenter=False,
    showCrosshair=True
)

cv2.destroyAllWindows()

if w == 0 or h == 0:
    print("No region selected")
else:
    # Convert display coordinates back to ORIGINAL image pixels
    bbox = [
        round(x / scale),
        round(y / scale),
        round((x + w) / scale),
        round((y + h) / scale),
    ]

    print("Original image size:", width, height)
    print("BBox [xmin, ymin, xmax, ymax]:", bbox)
