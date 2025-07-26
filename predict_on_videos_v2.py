import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

model = load_model('deepfake_cnn_model_v3.h5')
video_folder = "test_videos"
img_size = 224

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def preprocess_frame(frame):
    frame = cv2.resize(frame, (img_size, img_size))
    frame = img_to_array(frame) / 255.0
    return np.expand_dims(frame, axis=0)

for filename in os.listdir(video_folder):
    if not filename.endswith(".mp4"):
        continue
    video_path = os.path.join(video_folder, filename)
    frame = extract_middle_frame(video_path)
    if frame is None:
        print(f"❌ Could not extract frame from {filename}")
        continue

    processed = preprocess_frame(frame)
    pred = model.predict(processed, verbose=0)[0][0]

    if pred >= 0.6:
        label = "FAKE"
    elif pred <= 0.4:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    print("===================================")
    print(f"🎞️  Video: {filename}")
    print(f"🧠  Prediction: {label}")
    print(f"📊  Confidence: {pred:.4f}")
    print("===================================")
