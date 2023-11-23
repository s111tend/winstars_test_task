"""
@author: Daniel Hutsuliak
"""

# - Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.cluster import KMeans
from tqdm import tqdm

# Fucntion that build a mask for an image by the encoded pixel sequence
def create_mask(pixel_sequence, image_size=(768, 768)):
    image_length = image_size[0]*image_size[1] #define image length in pixels
    mask = np.zeros(image_length, dtype=np.bool) #create empty mask
    for start, length in pixel_sequence: #for each pair in sequence of index and length
        mask[start:start + length] = 1 #change colour for pixels from sequence

    return mask.reshape(image_size).transpose() #return reshaped transposed mask for image

# Function that encode a mask into pixel sequence
def encode_pixels(mask, threshold=0.5, image_size=(768, 768)):
    image_length = image_size[0]*image_size[1] #define image length in pixels
    row = (mask > threshold).astype(np.uint8).transpose().reshape(image_length) #use threshold, transpose and reshape to the mask
    
    pixel_sequence = '' #define empty string for encoded pixels
    counter = 0 #define counter variable
    
    for k, v in enumerate(row): #enumerate row
        if v == 1:
            #if value of pixel = 1 and counter = 0 add index of pixel to pixel_sequence
            if counter == 0:
                pixel_sequence += str(k)
            
            counter += 1 #increment counter
            
            #if value = 1 and it is the last pixel of image add '1' to pixel_sequence
            if k == image_length - 1:
                pixel_sequence += " 1 "
        else:
            #if value = 0 and counter != 0 add number of not-0 pixels to pixel_sequence and set counter to 0
            if counter != 0:
                pixel_sequence += f" {counter} "
                counter = 0
    
    return pixel_sequence[:-1] #return pixel_sequence without last space symbol (' ')

# Function that prepare train dataset for training a model
def prepare_train_dataset (train_dataset_path, train_images_path, train_size=-1, image_shape=(768, 768, 3), preprocessing_function=None):
    try:
        train = pd.read_csv(train_dataset_path) #read train dataset
        #if train_size is under 0 or it was not defined select all train images, else select limited amount
        if train_size <= 0:
            train_images_titles = os.listdir(train_images_path)
        else:
            train_images_titles = os.listdir(train_images_path)[:train_size]
    except:
        print("Incorrect values for dataset or images paths in 'prepare_train_dataset' function!")
        return 0

    train = train[train['ImageId'].isin(train_images_titles)] #crop train dataset if train_size > 0
    train['EncodedPixels'] = train['EncodedPixels'].fillna('No ships') #fill NaN values with string 'No ships'
    train['EncodedPixels'] = train['EncodedPixels'].apply(lambda x: str(x) + " " if x != 'No ships' else x) #add space to the and of EncodedPixels to concatenate it correctly
    train = train.groupby('ImageId').apply(sum) #groupe train dataset by image ids and concatenating EncodedPixels

    X = np.zeros((train_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.uint8) #create empty array for X
    Y = np.zeros((train_size, image_shape[0], image_shape[1], 1), dtype=np.bool) #create empty array for Y

    for i, image_title in tqdm(enumerate(train_images_titles)): #enumerate titles of train images
        img = imread(train_images_path + "/" + str(image_title))[:, :, :image_shape[2]] #read image
        #if shapes are different, resize it
        if img.shape != image_shape:
            img = resize(img, (image_shape[0], image_shape[1]), mode='constant', preserve_range=True)
        #if any preprocessing fucntion was defined, apply it on image
        if preprocessing_function != None:
            img = preprocessing_function(img)
        
        X[i] = img #add image to X array

        #create mask for image
        if train.loc[str(image_title)]['EncodedPixels'] != 'No ships':
            enc_pxls_array = np.array(train.loc[str(image_title)]['EncodedPixels'].split(' ')[:-1]).astype(int) #create array with index-length values of pixel sequences
            enc_pxls_array = enc_pxls_array.reshape((int(len(enc_pxls_array)/2), 2)) #reshape it for right 'create_mask' function work
            mask = create_mask(enc_pxls_array) #use this function to create image mask
            Y[i] = np.expand_dims(mask, axis=-1) #reshape to (height, width, 1)
        else:
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.bool) #create 0-mask for image without ships
            Y[i] = mask

    return X, Y

# Function that prepare test dataset for testing a model
def prepare_test_dataset (test_images_path, test_images_titles, image_shape=(768, 768, 3), test_size=-1, start_=0, preprocessing_function=None):
    try:
        #if test_size is positive int value, create an array of test images titles
        if test_size > 0:
            test_images_titles = test_images_titles[start_:start_ + test_size]
    except:
        print("Incorrect values for dataset or images paths in 'prepare_test_datasets' function!")
        return 0

    X = np.zeros((len(test_images_titles), image_shape[0], image_shape[1], image_shape[2]), dtype=np.uint8) #create empty array for X

    for i, image_title in tqdm(enumerate(test_images_titles)): #enumerate titles of test images
        img = imread(test_images_path + "/" + str(image_title))[:, :, :image_shape[2]] #read image
        #if shapes are different, resize it
        if img.shape != image_shape:
            img = resize(img, (image_shape[0], image_shape[1]), mode='constant', preserve_range=True)
        #if any preprocessing function was defined, apply it on image
        if preprocessing_function != None:
            img = preprocessing_function(img)

        X[i] = img #add image to X array

    return X

# Function that reduce a number of colours for image (preprocessing function)
def reduce_colours_number (image, n_colors=5):
    w, h, _ = image.shape #get image shape
    image = image.reshape(w*h, 3) #reshape image to (width, height, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image) #define kmeans clustering model and fit it
    labels = kmeans.predict(image) #predict clusters for each pixel
    identified_palette = np.array(kmeans.cluster_centers_).astype(int) #new palette (centroids of clusters)
    recolored_img = np.copy(image) #copy image
    for index in range(len(recolored_img)): #for each pixel
        recolored_img[index] = identified_palette[labels[index]] #recolor it using new palette
    recolored_img = recolored_img.reshape(w,h,3) #reshape it back

    return recolored_img