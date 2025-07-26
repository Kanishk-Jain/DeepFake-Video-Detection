# import tensorflow as tf
# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# #.\venv_gpu\Scripts\activate

from sklearn.utils import class_weight
import numpy as np

# Get class indices
labels = train_data.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
