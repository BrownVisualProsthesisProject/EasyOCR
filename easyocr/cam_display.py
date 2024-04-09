import depthai as dai
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
# Initialize TTS
classifier = load_model()

# Create pipeline
pipeline = dai.Pipeline()
# This might improve reducing the latency on some systems
pipeline.setXLinkChunkSize(0)

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setFps(30)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("out")
camRgb.isp.link(xout.input)
camRgb.setIspScale(1,2)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print(device.getUsbSpeed())
    q = device.getOutputQueue(name="out")
    while True:
        frameRgb = q.get().getCvFrame()
        cv2.imshow("framergb", cv2.resize(frameRgb, (0, 0), fx=.8, fy=.8))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Process the image using EasyOCR
            results_top = reader.readtext(frameRgb, slope_ths=.5, width_ths=.9, text_threshold=.5)
            # Combine the text of the bounding boxes to create one paragraph
            combined_text = ""
            for (bbox, text, prob) in results_top:
                combined_text += text + " "

            sentences = sent_tokenize(combined_text)
            
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
            cv2.imshow("res", cv2.resize(frameRgb, (0, 0), fx=.8, fy=.8))


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
                print(sentence)

            
    
        