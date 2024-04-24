import pygame
import cv2
import numpy as np
import sys
import sounddevice as sd
from nltk.tokenize import sent_tokenize
from easyocr import Reader
from TTS.api import TTS  # tts from coqui

def correct_skew(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur the image slightly to reduce high-frequency noise
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    # detect edges in the image
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    # use hough transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            # convert from radians to degrees and adjust
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)

        # compute the median angle of all detected lines
        median_angle = np.median(angles)
        # rotate the image around its center
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    return image

try:
    # Initialize Pygame for audio handling
    pygame.init()
    pygame.mixer.quit()
    pygame.mixer.init(frequency=22050, size=-16, channels=2)

    # Initialize EasyOCR reader
    reader = Reader(['en'], recog_network="english_g2")

    # Check if image path is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    # Read image path from command line argument
    image_path = sys.argv[1]

    # Read the image
    frameRgb = cv2.imread(image_path)
    if frameRgb is None:
        print("Error: Unable to load the image.")
        sys.exit(1)

    # Correct skew in the image
    corrected_image = correct_skew(frameRgb)

    # Process the image using EasyOCR
    results_top = reader.readtext(corrected_image, slope_ths=.5, width_ths=.9, text_threshold=.5)

    # Combine the text of the bounding boxes to create one paragraph
    combined_text = " ".join([text for (_, text, _) in results_top])

    # Tokenize the combined text into sentences using NLTK
    sentences = sent_tokenize(combined_text)

    # Initialize the TTS engine
    tts_engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

    # Process each sentence for speech synthesis
    for sentence in sentences:
        audio = tts_engine.tts(sentence)
        np_audio = np.array(audio, dtype=np.float32)
        # Convert float audio to int16 for playback through sounddevice
        int_audio = np.int16(np_audio * 32767)
        sd.play(int_audio, samplerate=22050)
        sd.wait()

    # Draw bounding boxes around the detected text
    for (bbox, text, prob) in results_top:
        # Extract bounding box coordinates
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        # Draw the bounding box on the corrected image
        cv2.rectangle(corrected_image, tl, br, (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow("Processed Image", cv2.resize(corrected_image, (0, 0), fx=1, fy=1))
    print("Combined Text:", combined_text)

    # Wait for a key press to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("\nExiting due to KeyboardInterrupt.")
    sys.exit(0)
