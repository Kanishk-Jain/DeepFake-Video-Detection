import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model just to initialize
model = load_model("deepfake_cnn_model_v2.h5")

# Rebuild data generators to re-run evaluate and fetch new history
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

img_height, img_width = 224, 224
batch_size = 32
dataset_dir = 'frames'

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Evaluate and simulate history for plotting
loss, accuracy = model.evaluate(val_data)

history_dict = {
    'val_accuracy': [accuracy],
    'val_loss': [loss]
}

# Save pseudo-history for plotting
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history_dict, f)

print(f"\n✅ History saved. Accuracy: {accuracy*100:.2f}%, Loss: {loss:.4f}")
