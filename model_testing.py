"""
@author: Daniel Hutsuliak
"""

# - Import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# - Import my own modules of this project
from modules.UNET import Unet
from modules.prepare import create_mask, prepare_test_dataset, reduce_colours_number, encode_pixels
from config.config import *

# - Initialize batches for testing model
batches = [
    [0, 1000], 
    [1000, 1000],
    [2000, 1000],
    [3000, 1000],
    [4000, 1000],
    [5000, 1000],
    [6000, 1000],
    [7000, 1000],
    [8000, 1000],
    [9000, 1000],
    [10000, 1000],
    [11000, 1000],
    [12000, 1000],
    [13000, 1000],
    [14000, 1000],
    [15000, 606]
]

# - Initialize variables
WEIGHTS_FILE_TITLE = 'model_weights_1.h5' #title of file with model's weights
SUBMISSIONS_FILE_NAME = 'submissions.csv' #title of resuting file

# - Initialize model
print("Initializing model...")
unet_nn = Unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
model = unet_nn.get_model()
model.load_weights(MODELS_PATH + "/" + str(WEIGHTS_FILE_TITLE)) #load trained model's weights
print("Done!")

pixel_sequence_array = [] #create empty array for pixel's sequences
image_ids = [] #create empty array for images' ids
test_images_titles = pd.read_csv('sample_submission_v2.csv')['ImageId'] #define list of titles of test images

for r, t in batches:
    # - Prepare test dataset
    print(f"Preparing test dataset...")
    X_test = prepare_test_dataset(
        TEST_IMAGES_PATH,
        test_images_titles,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        test_size=t,
        start_=r,
        preprocessing_function=None
    ) #use function to create test dataset
    print("Done!")

    # - Generate predictions for test dataset
    print("Predicting values for test dataset...")
    Y_pred = model.predict(X_test, workers=6) #predict test masks
    print("Done!")

    # - Prepare predictions:
    print("Preparing predicted values...")
    for k, v in tqdm(enumerate(Y_pred), total=len(Y_pred)): #for each predicted mask
        image_ids.append(test_images_titles[r + k]) #add image id to the array of ids
        pixel_sequence_array.append(encode_pixels(v, threshold=0.1)) #use encoding function and add result to the array of pixel sequences
    print("Done!")

# Creating resulting .csv file
print(f"Generating resulting file... (it will be saved in {RESULTS_PATH} folder)")
predictions = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': pixel_sequence_array}) #create dataframe of predictions
result[['ImageId', 'EncodedPixels']].to_csv(RESULTS_PATH + '/' + SUBMISSIONS_FILE_NAME, index=False) #save resulting dataset to .csv file
print("Done!")