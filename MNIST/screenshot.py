#!/usr/bin/env python3
import cv2
from mss import mss
from PIL import Image
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import callbacks

def show_img(im):
        plt.imshow(im, cmap='gray')
        plt.show()


bounding_box = {'top': 30, 'left': 859, 'width': 554, 'height': 851}
sct = mss()

# Defining the layers

checkpoint_path = "training_1/cp.ckpt"
model = Sequential(
    [
        # layers.Dropout(0.3),
        Dense(100, activation='relu', input_shape=[784]),
        # layers.Dropout(0.3),
        Dense(100, activation='relu'),
        # layers.Dropout(0.3),
        Dense(100, activation='relu'),
        Dense(10, activation='linear')
    ]
)
model.load_weights(checkpoint_path)
i = 0
while True:
    sct_img = sct.grab(bounding_box)
    image = np.array(sct_img)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_ = np.array([100, 50, 50]) 
    upper_ = np.array([140, 255, 255]) 
    mask = cv2.inRange(hsv, lower_, upper_)
    result = cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
    cv2.imwrite("img.png", roi)
    im = Image.open(r"img.png")
    im = im.resize((30,30)).crop((2, 2, 28, 28)).resize((28,28))
    im = im.convert("L")
    im = np.array(im).astype('float32')
    im = im.reshape(1,-1)
    y_a = model.predict(
        im,
        verbose = 0)
    y_an = tf.nn.softmax(y_a)
    y_ans = int(np.argmax(y_an))
    print("[",i,"] ->",y_ans)
    i+=1

    sleep(3)