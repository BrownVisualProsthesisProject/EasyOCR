import pygame
import cv2
from easyocr import Reader
import numpy as np
import sys
from nltk.tokenize import sent_tokenize
import requests
from TTS import load_model

# Initialize Pygame
pygame.init()
pygame.mixer.quit()
pygame.mixer.init(16000, -16, 2)


url = "http://localhost:8000/predict"
headers = {"accept": "application/json"}

# Initialize EasyOCR reader
reader = Reader(['en'], recog_network="english_g2")

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
results_top = reader.readtext(frameRgb, slope_ths=.5, width_ths=.9, text_threshold=.5)

# Combine the text of the bounding boxes to create one paragraph
combined_text = ""
for (bbox, text, prob) in results_top:
    combined_text += text + " "

sentences = sent_tokenize(combined_text)
classifier = load_model()

for sentence in sentences:
    # Pause for 2 seconds before speaking each sentence
    #pygame.time.wait(10)
    #params = {"text": sentence}
    #response = requests.post(url, params=params, headers=headers)
    #data = response.json()
    #npa = np.asarray(data['data'], dtype=np.int16)
    npa, sample_rate = classifier(sentence) 
    npa = np.repeat(npa.reshape(len(npa), 1), 2, axis = 1)
    # Play the audio
    sound = pygame.sndarray.make_sound(npa)
    sound.play()
    pygame.time.wait(int(sound.get_length() * 1000))

# Draw bounding boxes around the detected text
for (bbox, text, prob) in results_top:
    # Extract bounding box coordinates
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    # Draw bounding box
    cv2.rectangle(frameRgb, tl, br, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow("framergb", cv2.resize(frameRgb, (0, 0), fx=1, fy=1))
print("Combined Text:", combined_text)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()

