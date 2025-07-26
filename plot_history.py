import matplotlib.pyplot as plt
import pickle

# Load saved history
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plotting
plt.figure(figsize=(10, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['val_accuracy'], marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['val_loss'], marker='o', color='red')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig("training_plots.png")  # Save the plot as an image
plt.show()
