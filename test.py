import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import threading
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from collections import deque


class SignLanguageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="#f0f0f0")
        self.window.geometry("1280x720")

        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        # Increase maxFaces for better hand detection reliability
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

        # Enhanced parameters
        self.offset = 25  # Increased offset for better hand capture
        self.imgSize = 300
        self.is_running = True

        # Labels for ASL signs
        self.labels = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "del", "nothing", "space"
        ]

        # Prediction stabilization with rolling window
        self.prediction_history = deque(maxlen=7)  # Stores recent predictions
        self.confidence_threshold = 0.75  # Minimum confidence to consider a prediction
        self.stable_threshold = 5  # Number of same predictions needed to consider it stable

        # Create main frames
        self.main_frame = tk.Frame(window, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.frame_left = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.frame_left.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.frame_right = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Video display
        self.video_frame = tk.Label(self.frame_left)
        self.video_frame.pack(padx=10, pady=10)

        # Debug info - shows processed hand image
        self.debug_frame = tk.LabelFrame(self.frame_left, text="Processed Hand", font=("Arial", 12), bg="#f0f0f0")
        self.debug_frame.pack(padx=10, pady=10, fill="x")

        self.debug_display = tk.Label(self.debug_frame)
        self.debug_display.pack(padx=10, pady=10)

        # Prediction confidence visualization
        self.confidence_frame = tk.LabelFrame(self.frame_left, text="Prediction Confidence", font=("Arial", 12),
                                              bg="#f0f0f0")
        self.confidence_frame.pack(padx=10, pady=10, fill="x")

        self.confidence_bar = ttk.Progressbar(self.confidence_frame, orient="horizontal", length=200,
                                              mode="determinate")
        self.confidence_bar.pack(padx=10, pady=10, fill="x")

        # Prediction display
        self.pred_frame = tk.LabelFrame(self.frame_right, text="Current Prediction", font=("Arial", 14), bg="#f0f0f0")
        self.pred_frame.pack(padx=10, pady=10, fill="x")

        self.prediction_label = tk.Label(self.pred_frame, text="", font=("Arial", 120, "bold"), bg="#f0f0f0")
        self.prediction_label.pack(padx=10, pady=10)

        # Accuracy display
        self.accuracy_label = tk.Label(self.pred_frame, text="Accuracy: 0.00%", font=("Arial", 16), bg="#f0f0f0")
        self.accuracy_label.pack(padx=10, pady=5)

        # Stability indicator
        self.stability_label = tk.Label(self.pred_frame, text="Stability: 0/5", font=("Arial", 14), bg="#f0f0f0")
        self.stability_label.pack(padx=10, pady=5)

        # Text display (accumulated predictions)
        self.text_frame = tk.LabelFrame(self.frame_right, text="Recognized Text", font=("Arial", 14), bg="#f0f0f0")
        self.text_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.text_display = scrolledtext.ScrolledText(self.text_frame, height=5, width=30, font=("Arial", 16),
                                                      wrap=tk.WORD)
        self.text_display.pack(padx=10, pady=10, fill="both", expand=True)

        # Buttons
        self.button_frame = tk.Frame(self.frame_right, bg="#f0f0f0")
        self.button_frame.pack(padx=10, pady=10, fill="x")

        self.clear_button = tk.Button(self.button_frame, text="Clear Text", width=15,
                                      command=self.clear_text, bg="#f0e68c")
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = tk.Button(self.button_frame, text="Save to File", width=15,
                                     command=self.save_text, bg="#90ee90")
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.quit_button = tk.Button(self.button_frame, text="Quit", width=15,
                                     command=self.quit_app, bg="#ff6347")
        self.quit_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Status bar
        self.status_bar = tk.Label(self.window, text="Ready. Position your hand in frame to begin.",
                                   bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Variables for prediction stability
        self.current_pred = "nothing"
        self.current_conf = 0
        self.pred_count = 0
        self.processed_hand = None
        self.last_added_text = None
        self.add_text_cooldown = 0  # Prevents text spam

        # Start video processing thread
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.window.mainloop()

    def process_video(self):
        while self.is_running:
            success, img = self.cap.read()
            if not success:
                continue

            # Flip the image horizontally for a more intuitive experience
            img = cv2.flip(img, 1)
            imgOutput = img.copy()

            # Create a white image for hand processing
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255

            # Detect hands
            hands, img = self.detector.findHands(img)

            # Initialize prediction variables
            prediction = np.zeros(len(self.labels))
            index = -1
            max_conf = 0
            max_index = 0

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Apply contrast enhancement to hand region if possible
                if (x - self.offset >= 0 and y - self.offset >= 0 and
                        x + w + self.offset <= img.shape[1] and y + h + self.offset <= img.shape[0]):

                    # Extract hand region with padding
                    imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                    if imgCrop.size != 0:  # Check if the crop is valid
                        # Apply preprocessing to enhance features
                        imgCrop = self.preprocess_hand(imgCrop)

                        # Calculate aspect ratio for proper resizing
                        aspectRatio = h / w

                        if aspectRatio > 1:
                            k = self.imgSize / h
                            wCal = math.ceil(k * w)
                            if wCal > 0:  # Ensure width is positive
                                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                                wGap = math.ceil((self.imgSize - wCal) / 2)
                                imgWhite[:, wGap:wCal + wGap] = imgResize
                                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                        else:
                            k = self.imgSize / w
                            hCal = math.ceil(k * h)
                            if hCal > 0:  # Ensure height is positive
                                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                                hGap = math.ceil((self.imgSize - hCal) / 2)
                                imgWhite[hGap:hCal + hGap, :] = imgResize
                                prediction, index = self.classifier.getPrediction(imgWhite, draw=False)

                        # Store processed hand image for display
                        self.processed_hand = imgWhite.copy()

                        # Get the highest confidence prediction
                        max_conf = np.max(prediction)
                        max_index = np.argmax(prediction)

                        # Update the prediction history queue
                        if max_conf > self.confidence_threshold:
                            self.prediction_history.append(max_index)

                            # Count occurrences of most common prediction in history
                            pred_counts = {}
                            for p in self.prediction_history:
                                if p in pred_counts:
                                    pred_counts[p] += 1
                                else:
                                    pred_counts[p] = 1

                            # Find most common prediction and its count
                            most_common_pred = max(pred_counts, key=pred_counts.get, default=-1)
                            stability_count = pred_counts.get(most_common_pred, 0)

                            # Update stability indicator
                            self.window.after(0, self.update_stability, stability_count)

                            # If prediction is stable, update display
                            if stability_count >= self.stable_threshold and most_common_pred == max_index:
                                label_text = self.labels[max_index]
                                self.window.after(0, self.update_prediction, label_text, max_conf * 100)

                                # If this is a new prediction, add to text with cooldown
                                if self.last_added_text != label_text and self.add_text_cooldown <= 0:
                                    self.window.after(0, self.add_to_text, label_text)
                                    self.last_added_text = label_text
                                    self.add_text_cooldown = 15  # Set cooldown frames

                                    # Clear history to prevent immediate repetition
                                    self.prediction_history.clear()

                        # Update the confidence bar
                        self.window.after(0, self.update_confidence_bar, max_conf * 100)

                        # Update status
                        self.window.after(0, self.update_status,
                                          f"Detected: {self.labels[max_index]} ({max_conf * 100:.1f}%)")

                        # Display visual feedback on the video
                        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 50),
                                      (x - self.offset + 120, y - self.offset), (255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, self.labels[max_index], (x - self.offset + 10, y - self.offset - 10),
                                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset),
                                      (x + w + self.offset, y + h + self.offset), (255, 0, 255), 4)
            else:
                self.window.after(0, self.update_status, "No hand detected. Position your hand in frame.")
                # Clear prediction history when no hand detected
                self.prediction_history.clear()
                self.window.after(0, self.update_stability, 0)
                self.window.after(0, self.update_confidence_bar, 0)

            # Decrement cooldown counter
            if self.add_text_cooldown > 0:
                self.add_text_cooldown -= 1

            # Convert to RGB for display
            imgOutput_rgb = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
            imgOutput_pil = Image.fromarray(imgOutput_rgb)

            # Resize for display
            width, height = 640, 480
            imgOutput_pil = imgOutput_pil.resize((width, height), Image.LANCZOS)

            # Convert to PhotoImage
            imgtk = ImageTk.PhotoImage(image=imgOutput_pil)

            # Update video frame
            self.window.after(0, self.update_video, imgtk)

            # Update debug display if we have a processed hand
            if self.processed_hand is not None:
                processed_rgb = cv2.cvtColor(self.processed_hand, cv2.COLOR_BGR2RGB)
                processed_pil = Image.fromarray(processed_rgb)

                # Resize for display (smaller than main video)
                processed_pil = processed_pil.resize((200, 200), Image.LANCZOS)

                # Convert to PhotoImage
                processed_tk = ImageTk.PhotoImage(image=processed_pil)

                # Update debug display
                self.window.after(0, self.update_debug, processed_tk)

    def preprocess_hand(self, img):
        """Apply image preprocessing to enhance hand features"""
        # Convert to grayscale and back to BGR for better feature extraction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Apply slight Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Enhance edges
        edges = cv2.Canny(gray, 50, 150)

        # Convert back to BGR
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return enhanced

    def update_video(self, img):
        self.video_frame.configure(image=img)
        self.video_frame.image = img

    def update_debug(self, img):
        self.debug_display.configure(image=img)
        self.debug_display.image = img

    def update_status(self, text):
        self.status_bar.config(text=text)

    def update_confidence_bar(self, value):
        self.confidence_bar["value"] = value

    def update_stability(self, count):
        self.stability_label.config(text=f"Stability: {count}/{self.stable_threshold}")

        # Change color based on stability level
        if count >= self.stable_threshold:
            self.stability_label.config(fg="green")
        elif count >= self.stable_threshold / 2:
            self.stability_label.config(fg="orange")
        else:
            self.stability_label.config(fg="red")

    def update_prediction(self, pred_text, accuracy):
        self.prediction_label.config(text=pred_text)
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")

    def add_to_text(self, pred_text):
        """Add prediction to text display with special character handling"""
        if pred_text == "space":
            self.text_display.insert(tk.END, " ")
        elif pred_text == "del":
            # Delete the last character
            current_text = self.text_display.get("1.0", tk.END)[:-1]  # Remove the automatic newline
            if current_text:
                self.text_display.delete("1.0", tk.END)
                self.text_display.insert(tk.END, current_text[:-1])
        elif pred_text not in ["nothing"]:
            self.text_display.insert(tk.END, pred_text)

        self.text_display.see(tk.END)  # Scroll to the end

    def clear_text(self):
        self.text_display.delete("1.0", tk.END)
        self.last_added_text = None  # Reset last added text

    def save_text(self):
        text = self.text_display.get("1.0", tk.END).strip()
        if text:
            with open("sign_language_text.txt", "w") as file:
                file.write(text)
            # Give user feedback
            status_window = tk.Toplevel(self.window)
            status_window.title("Save Status")
            tk.Label(status_window, text="Text saved successfully to sign_language_text.txt",
                     padx=20, pady=10).pack()
            tk.Button(status_window, text="OK", command=status_window.destroy).pack(pady=10)

    def quit_app(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    # Create Tkinter window
    root = tk.Tk()
    # Import ttk for progress bar
    from tkinter import ttk

    app = SignLanguageApp(root, "Sign Language Recognition System")