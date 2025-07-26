This project is a complete pipeline to detect whether a given video is real or AI-generated (deepfake). It uses a custom-trained Convolutional Neural Network (CNN) model to analyze extracted video frames and classify the video with a real/fake label and confidence score.

-> Project Highlights
1. Developed a deep learning model to classify deepfake vs. real video content
2. Automated frame extraction from .mp4 videos using OpenCV
3. Flask web interface to upload videos and view predictions in real-time
4. Plots for model evaluation and training curves
5. Batch prediction support for testing multiple videos at once
6. Optimized and cleaned for GitHub, with heavy files managed separately

-> Features
1. Frame Extraction: Extracts consistent frames from uploaded videos
2. Custom CNN Model: Trained on labeled deepfake and real video frames
3. Inference Pipeline: Predicts label using average of per-frame probabilities
4. Web UI: Flask-based upload interface with prediction output
5. Evaluation Scripts: ROC curve, accuracy, loss, and training plot generation
6. Batch Mode: Automatically processes multiple videos for testing


deepfake-detector/

├── app.py                       # Flask app for video upload and prediction

├── train_model_v3.py            # Model training script (CNN)

├── test_videos_batch.py         # Batch prediction on test videos

├── extract_frames.py            # Extracts frames from a video file

├── evaluate_model.py            # Evaluation metrics and plots

├── model/                       # (Optional) Folder for .h5 file (model)

├── static/                      # For CSS, spinner, etc.

├── templates/index.html         # Frontend template for Flask

├── .gitignore                   # Ignores venv, .h5, frames/, etc.

└── README.md


Input: Preprocessed 224x224 RGB video frames
Architecture: Custom CNN with Conv2D, MaxPooling, Dropout, Dense layers
Output: Binary classification (Real vs. Fake)
Evaluation: Accuracy, Loss, ROC AUC, Precision, Recall

**This project is for educational and research purposes only. Not intended for commercial use.**
