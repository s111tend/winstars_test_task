"""
@author: Daniel Hutsuliak
"""

IMAGE_WIDTH = 768 #image width
IMAGE_HEIGHT = 768 #image height
IMAGE_CHANNELS = 3 #image depth
TRAIN_DATASET_PATH = 'dataset/train_ship_segmentations_v2.csv' #path to folder with train dataset
SUBM_FILE_PATH = 'dataset/sample_submission_v2.csv' #path and title of submission example file
TRAIN_IMAGES_PATH = 'dataset/train_v2' #path to folder with images for model training
TEST_IMAGES_PATH = 'dataset/test_v2' #path to folder with images for model testing
MODELS_PATH = 'models' #path to folder with saved weights for models
RESULTS_PATH = 'results' #path to folder where we save results