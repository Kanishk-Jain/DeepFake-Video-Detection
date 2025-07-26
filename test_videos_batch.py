import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("deepfake_cnn_model_v3.h5")

# Parameters
img_size = 224
video_dir = "test_videos"

# Function to preprocess a frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype("float32") / 255.0
    return frame

# Iterate through all videos
for filename in os.listdir(video_dir):
    if not filename.endswith(('.mp4', '.avi', '.mov')):
        continue

    video_path = os.path.join(video_dir, filename)
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 == 0:  # take every 10th frame
            frames.append(preprocess_frame(frame))
    cap.release()

    if len(frames) == 0:
        print(f"{filename}: ❌ No valid frames found.")
        continue

    input_batch = np.array(frames)
    preds = model.predict(input_batch)
    mean_pred = np.mean(preds)

    label = "REAL" if mean_pred >= 0.5 else "FAKE"
    confidence = 1 - mean_pred if label == "FAKE" else mean_pred

    print("===================================")
    print(f"🎞️  Video: {filename}")
    print(f"🧠  Prediction: {label}")
    print(f"📊  Confidence: {confidence:.4f}")
    print("===================================")
