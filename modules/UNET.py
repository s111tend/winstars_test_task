"""
@author: Daniel Hutsuliak
"""

# Import libraries
import tensorflow as tf

# Define class Unet
class Unet:
    def __init__(self, image_height=768, image_width=768, image_channels=3):
        print("Initializing vars for U-net model...")
        self.im_height = image_height
        self.im_width = image_width
        self.im_channels = image_channels
        print("Done!")

    def get_model(self):
        inputs = tf.keras.layers.Input(shape = (self.im_height, self.im_width, self.im_channels)) #define inputs
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) #normalize it

        # Contraction path
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s) #convolutional layer
        c1 = tf.keras.layers.Dropout(0.1)(c1) #dropout layer
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1) #convolutional layer
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) #max pooling layer

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1) #convolutional layer
        c2 = tf.keras.layers.Dropout(0.1)(c2) #dropout layer
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2) #convolutional layer
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) #max pooling layer
         
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2) #convolutional layer
        c3 = tf.keras.layers.Dropout(0.2)(c3) #dropout layer
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3) #convolutional layer
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3) #max pooling layer
         
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3) #convolutional layer
        c4 = tf.keras.layers.Dropout(0.2)(c4) #dropout layer
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4) #convolutional layer
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4) #max pooling layer
         
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4) #convolutional layer
        c5 = tf.keras.layers.Dropout(0.3)(c5) #dropout layer
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5) #convolutional layer

        # Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) #deconvolutional layer
        u6 = tf.keras.layers.concatenate([u6, c4]) #concatenating
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6) #convolutional layer
        c6 = tf.keras.layers.Dropout(0.2)(c6) #dropout layer
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6) #convolutional layer
         
        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6) #deconvolutional layer
        u7 = tf.keras.layers.concatenate([u7, c3]) #concatenating
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7) #convolutional layer
        c7 = tf.keras.layers.Dropout(0.2)(c7) #dropout layer
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7) #convolutional layer
         
        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) #deconvolutional layer
        u8 = tf.keras.layers.concatenate([u8, c2]) #concatenating
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8) #convolutional layer
        c8 = tf.keras.layers.Dropout(0.1)(c8) #dropout layer
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8) #convolutional layer
         
        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8) #deconvolutional layer
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3) #concatenating
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9) #convolutional layer
        c9 = tf.keras.layers.Dropout(0.1)(c9) #dropout layer
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9) #convolutional layer
         
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9) #define outputs with final convolutional layer
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs]) #initialize model
        
        return model