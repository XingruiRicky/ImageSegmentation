import os
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers


input_dir = "../input/oxford-pets/images/images"
target_dir = "../input/oxford-pets/annotations/annotations/trimaps"
img_size = (160, 160)
num_classes = 3
batch_size = 16

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

# Display image #7
i = 7
figure, ax = plt.subplots(nrows=1,ncols=2,figsize=(8,8))
ax.ravel()[0].imshow(mpimg.imread(input_img_paths[i]))
ax.ravel()[0].set_title("Orginal image")
ax.ravel()[0].set_axis_off()
ax.ravel()[1].imshow(mpimg.imread(target_img_paths[i]))
ax.ravel()[1].set_title("Mask")
ax.ravel()[1].set_axis_off()
#ax.ravel()[2].imshow(PIL.ImageOps.autocontrast(load_img(target_img_paths[i])))
#ax.ravel()[2].set_title("Contrast of mask")
#ax.ravel()[2].set_axis_off()
plt.tight_layout()

class PetsDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y
    
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()

val_samples = 1108 # 85% Training -- 15% Validation
random.Random(1822).shuffle(input_img_paths)
random.Random(1822).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = PetsDataset(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = PetsDataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)

# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint("pets_segmentation.h5", save_best_only=True)
]

epochs = 30
modelunet=model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

# summarize history for accuracy
plt.plot(modelunet.history['accuracy'])
plt.plot(modelunet.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.grid(True)
plt.show()
# summarize history for loss
plt.plot(modelunet.history['loss'])
plt.plot(modelunet.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.grid(True)
plt.show()

# Generate predictions for all images in the validation set
val_gen = PetsDataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)

def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    return img

# Display image #120
i = 120
figure, ax = plt.subplots(nrows=1,ncols=3,figsize=(8,5))
ax.ravel()[0].imshow(mpimg.imread(val_input_img_paths[i]))
ax.ravel()[0].set_title("Orginal image")
ax.ravel()[0].set_axis_off()
ax.ravel()[1].imshow(mpimg.imread(val_target_img_paths[i]))
ax.ravel()[1].set_title("Mask")
ax.ravel()[1].set_axis_off()
ax.ravel()[2].imshow(display_mask(i))
ax.ravel()[2].set_title("Predicted mask ")
ax.ravel()[2].set_axis_off()
plt.tight_layout()

# Display image #180
i = 180
figure, ax = plt.subplots(nrows=1,ncols=3,figsize=(8,5))
ax.ravel()[0].imshow(mpimg.imread(val_input_img_paths[i]))
ax.ravel()[0].set_title("Orginal image")
ax.ravel()[0].set_axis_off()
ax.ravel()[1].imshow(mpimg.imread(val_target_img_paths[i]))
ax.ravel()[1].set_title("Mask")
ax.ravel()[1].set_axis_off()
ax.ravel()[2].imshow(display_mask(i))
ax.ravel()[2].set_title("Predicted mask ")
ax.ravel()[2].set_axis_off()
plt.tight_layout()