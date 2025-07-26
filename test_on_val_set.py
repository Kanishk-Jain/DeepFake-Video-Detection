import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load the trained model
model = load_model("deepfake_cnn_model_v2.h5")

# Image configuration
img_height, img_width = 224, 224
batch_size = 32
dataset_dir = "frames"

# Create validation generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Predict probabilities
pred_probs = model.predict(val_generator)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

# True labels
true_labels = val_generator.classes
file_paths = val_generator.filepaths

# Print predictions
for i in range(len(file_paths)):
    fname = os.path.basename(file_paths[i])
    actual = "Real" if true_labels[i] == 0 else "DeepFake"
    predicted = "Real" if pred_labels[i] == 0 else "DeepFake"
    confidence = pred_probs[i][0]
    print(f"{fname} | Actual: {actual} | Predicted: {predicted} | Confidence: {confidence:.2f}")
