import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("deepfake_cnn_model_v3.h5")
img_size = 224

def preprocess_frame(frame):
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype("float32") / 255.0
    return frame

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 == 0:
            frames.append(preprocess_frame(frame))
    cap.release()

    if not frames:
        return "INVALID", 0.0

    input_batch = np.array(frames)
    preds = model.predict(input_batch)
    mean_pred = np.mean(preds)

    label = "REAL" if mean_pred >= 0.5 else "FAKE"
    confidence = float(mean_pred if label == "REAL" else 1 - mean_pred) * 100
    return label, round(confidence, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        if "video" not in request.files:
            return render_template("index.html", result="No video uploaded", confidence=0)

        file = request.files["video"]
        if file.filename == "":
            return render_template("index.html", result="No selected file", confidence=0)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        result, confidence = predict_video(filepath)

        os.remove(filepath)

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
