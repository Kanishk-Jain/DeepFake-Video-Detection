import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = load_model("deepfake_cnn_model_v2.h5")

# Image settings
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

# Predict
pred_probs = model.predict(val_generator)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = val_generator.classes

# 📌 Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
class_names = list(val_generator.class_indices.keys())

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 📌 Classification Report
report = classification_report(true_labels, pred_labels, target_names=class_names)
print("\n🧾 Classification Report:\n")
print(report)
