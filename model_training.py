"""
@author: Daniel Hutsuliak
"""

# - Import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import os

# - Import my own modules of this project
from modules.UNET import Unet
from modules.prepare import create_mask, prepare_train_dataset, reduce_colours_number
from config.config import *

# - Initialize variables
TRAIN_SIZE = 5000

# - Prepare train dataset
print(f"Preparing train dataset: size={TRAIN_SIZE}...")
X_train, Y_train = prepare_train_dataset(
    TRAIN_DATASET_PATH, 
    TRAIN_IMAGES_PATH, 
    train_size=TRAIN_SIZE, 
    image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
    preprocessing_function=None
) #use function to create train dataset
print("Done!")

# - Initialize model
print("Initializing model...")
unet_nn = Unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
model = unet_nn.get_model()
print("Done!")

# - Compile model and printing it's summary
print("Compiling model...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #compile model
model.summary() #print info about our model
print("Done!")

# - Create model callbacks
print("Creating callbacks...")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODELS_PATH + '/model_weights_1.h5', verbose=1, save_best_only=True), #checkpointer
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), #early stopping callback
    tf.keras.callbacks.TensorBoard(log_dir='logs') #tensorboad callback for saving info about training process
]
print("Done!")

# Fit model
print("Fitting model...")
results = model.fit(
    X_train, 
    Y_train, 
    validation_split=0.1, 
    batch_size=16, 
    epochs=20, 
    callbacks=callbacks, 
    workers=6
) #fit model
print("Done!")