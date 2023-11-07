import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization

training_images_path = "/Users/utkarsh/Downloads/MNIST/train-images-idx3-ubyte.gz"
training_labels_path = "/Users/utkarsh/Downloads/MNIST/train-labels-idx1-ubyte.gz"
 
train_images_byte = gzip.open(training_images_path,'r')
image_size = 28
sample_size = 60000 
train_images_byte.read(16)
buf = train_images_byte.read(image_size * image_size * sample_size)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
images = data.reshape(sample_size, image_size, image_size, 1)

train_labels_byte = gzip.open(training_labels_path,'r')
train_labels_byte.read(8)
buf = train_labels_byte.read(sample_size)
labels = np.frombuffer(buf, dtype=np.uint8)

def show_images(images, labels):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(10,7))
    index = 1   
    for image, label in zip(images, labels):        
        plt.subplot(rows, cols, index)        
        plt.imshow(image.squeeze(), cmap=plt.cm.gray)
        plt.title(f'Label: {label}')
        plt.axis('off')       
        index += 1

X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.2)

early_stopping = tf.keras.callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)

model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Conv2D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.0005),
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=500,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

testing_images_path = "/Users/utkarsh/Downloads/MNIST/t10k-images-idx3-ubyte.gz"
testing_labels_path = "/Users/utkarsh/Downloads/MNIST/t10k-labels-idx1-ubyte.gz"
test_images_byte = gzip.open(testing_images_path,'r')
test_images_byte.read(16)
buf = test_images_byte.read(image_size * image_size * sample_size)
test_images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
test_images = test_images.reshape(sample_size, image_size, image_size, 1)

test_labels_byte = gzip.open(testing_labels_path,'r')
test_labels_byte.read(8)
buf = test_labels_byte.read(sample_size)
test_labels = np.frombuffer(buf, dtype=np.uint8)

X_test = test_images
y_test = test_labels

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

accuracy = np.mean(y_pred_labels == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
