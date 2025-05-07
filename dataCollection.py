import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Dataset/space"
counter = 0

# Ensure folder exists
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera")
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Add boundary checks to prevent cropping outside image
        imgHeight, imgWidth, _ = img.shape

        # Calculate safe crop coordinates
        y1 = max(0, y - offset)
        y2 = min(imgHeight, y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(imgWidth, x + w + offset)

        # Only proceed if we have a valid crop region
        if y2 > y1 and x2 > x1:
            imgCrop = img[y1:y2, x1:x2]

            # Create a white background image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Check if imgCrop is not empty
            if imgCrop.size != 0:
                aspectRatio = (y2 - y1) / (x2 - x1)

                if aspectRatio > 1:
                    # Height is greater than width
                    k = imgSize / (y2 - y1)
                    wCal = math.ceil(k * (x2 - x1))
                    if wCal > 0:
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    # Width is greater than height
                    k = imgSize / (x2 - x1)
                    hCal = math.ceil(k * (y2 - y1))
                    if hCal > 0:
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(f"Saved: {counter}")
            else:
                print("Warning: Empty crop detected - hand may be too close to edge of frame")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # Exit if ESC key is pressed
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()