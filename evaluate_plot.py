import matplotlib.pyplot as plt
import pickle

# Step 1: Load history object (jo pickle file me save kiya tha)
with open('full_training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Step 2: Plot Accuracy and Loss
epochs = range(1, len(history['accuracy']) + 1)

plt.figure(figsize=(14, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, history['accuracy'], 'b', label='Training Accuracy')
plt.plot(epochs, history['val_accuracy'], 'g', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, history['loss'], 'r', label='Training Loss')
plt.plot(epochs, history['val_loss'], 'orange', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
