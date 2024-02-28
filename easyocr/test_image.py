import cv2
from easyocr import Reader
import numpy as np
import sys

# Initialize EasyOCR reader
reader = Reader(['en'])

# Check if image path is provided as argument
if len(sys.argv) != 2:
    print("Usage: python script.py <image_path>")
    sys.exit(1)

# Read image path from command line argument
image_path = sys.argv[1]

# Read the image
frameRgb = cv2.imread(image_path)

# Check if the image is loaded successfully
if frameRgb is None:
    print("Error: Unable to load the image.")
    sys.exit(1)

# Process the image using EasyOCR
results_top = reader.readtext(frameRgb, paragraph=True, y_ths=.1)

# Display the image
cv2.imshow("framergb", cv2.resize(frameRgb, (0, 0), fx=0.5, fy=0.5))
print(results_top)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
