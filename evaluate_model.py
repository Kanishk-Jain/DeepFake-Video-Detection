import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = load_model("deepfake_cnn_model_v2.h5")

# Validation data generator
img_height, img_width = 224, 224
batch_size = 32
dataset_dir = 'frames'

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False  # IMPORTANT for correct predictions alignment
)

# Predict
pred_probs = model.predict(val_data)
pred_labels = (pred_probs > 0.5).astype(int).reshape(-1)
true_labels = val_data.classes

# 1️⃣ Confusion Matrix
print("\n📊 Confusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))

# 2️⃣ Classification Report
print("\n📋 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=val_data.class_indices.keys()))

# 3️⃣ ROC Curve
fpr, tpr, _ = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
