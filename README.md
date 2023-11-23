##### Author: Daniel Hutsuliak

# Introduction
This assignment was a real challenge for me. I had never worked with computer vision technology before, and I had to study a lot of material on the subject. Most of them are listed in the "resources" section at the bottom of this file. 
Overall, it was very interesting and I learned a lot in the process of completing this assignment. I have tried to describe and organize the results of my work in a way that is as easy to study as possible.

# Task
Current task requires solving one of the problems from the Kaggle platform: [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection/overview).
The goal of the test project is to build a semantic segmentation model. Prefered tools and notes: tf.keras, Unet architecture for neural network, dice score, python. 
### Results of work should contain next:
- link to GitHub repository with all source codes;
- code for model training and model inference should be separated into different .py files;
- readme.md file with complete description of solution;
- requirements.txt with required python modules;
- jupyter notebook with exploratory data analysis of the dataset;
- any other things used during the working with task;
### Requirements:
- Source code should be well readable and commented;
- Project has to be easy to deploy for testing;

# Project structure
The project has the following structure:
- 'config' folder, containing files with parameters on the type of paths to different folders and files, image sizes etc
- 'dataset' folder with data sets
- 'img' folder with images
- 'models' folder with weights of models obtained as a result of neural network training
- 'modules' folder with files containing data processing functions, model classes, etc
- 'results' folder for resulting .csv files to submit them on kaggle
- files and notebooks for model training and testing (.py files are required by condition of the task, but I recommend you to see the notebooks with results of training and testing models for greater clarity)

# Solution overview
So, according to the terms of the assignment, we need to implement training of the U-net neural network on the provided data set using the methods of tensorflow framework, test the obtained model and check its quality by evaluating it with the help of kaggle platform.
### U-net architecture:
![U-net NN](img/U-net-architecture.png)
U-Net is a convolutional neural network (CNN) architecture designed for semantic image segmentation. The name "U-Net" comes from the U-shaped architecture of the network. The U-Net architecture consists of two main parts: the contracting path (encoder) and the expansive path (decoder). The architecture is symmetrical, and the contracting and expansive paths are connected by a bottleneck in the middle.
1. Contracting Path (Encoder):
The contracting path is responsible for capturing the context and extracting features from the input image.
It consists of a series of convolutional layers followed by rectified linear unit (ReLU) activation functions and max-pooling operations.
These operations progressively reduce the spatial dimensions of the input image while increasing the number of feature channels.
2. Bottleneck:
The bottleneck serves as a bridge between the contracting and expansive paths.
It typically consists of a stack of convolutional layers with ReLU activations but without max-pooling.
This part captures the most abstract and high-level features.
3. Expansive Path (Decoder):
The expansive path is responsible for upsampling the feature maps and generating the segmentation mask.
It consists of up-convolutional layers (also known as transposed convolutions or deconvolutions) to increase spatial resolution.
Each up-convolutional layer is followed by concatenation with the corresponding feature maps from the contracting path (skip connections).
After the concatenation, the feature maps go through a series of convolutional layers and ReLU activations.
4. Final Layer:
The final layer typically consists of a 1x1 convolutional layer with a sigmoid activation function.
The output of the final layer represents the segmentation mask, where each pixel indicates the probability of belonging to a particular class.
### Data analysis and data preparation:
Before we get started, we need to inspect and explore the dataset on which we will train our neural network, as well as create a functional tool to prepare the data for use, taking into account any possible problems that may arise with the original data. For example, if the original images have different shapes, when training the data we need to make sure that these images are reduced to the same size. We will perform all the above procedures in two files of our project: `Dataset Analysis.ipynb` for researching the initial data and planning the work of the tools for preparing the data for work and the file `modules/prepare.py`, which will contain full-fledged functions for data processing.
### Modeling:
To describe the structure of our neural network, we will create a separated file where we will describe it as a class to make it easier to test our model in the future, because according to the condition of the assignment, training and testing of the model should be described in two different files. For this purpose, we will create a folder `models` for storing the weights of our neural networks. Then it will be easy to create an object of the model class, initialize weights for it and use it for predictions on the test dataset.
### Results:
We will eventually present the results, numerical and graphical in the final notebook: `Visual model testing.ipynb`. The final validation file (`results/submission.csv`) should contain 17004 lines. For each image, if there is more than one ship on it, it is necessary to separate them into several different lines. That is, for one image there can be more than one line in the final file. Honestly, I don't know how to do this, provided that our model is extremely weak for predicting the number of ships in an image. So the code that creates the submission.csv file is not correct. It creates one mask for each image. And since the kaggle check requires a file with a strictly fixed number of lines, it will not be possible to predict all this correctly with our model. In spite of this, I think that I coped with the task, because all the remaining problems and inaccuracies did not depend on me: I do not have access to sufficient computational resources to create a better model.

# Installation
### Installation of project files
All necessary project files except for the dataset itself are provided in the main directory. The datasets are too large to upload to github - they weigh about 30 gigabytes. Therefore, if you want to check the work of the models personally, you need to follow the [link](https://www.kaggle.com/competitions/airbus-ship-detection/data) and download the dataset yourself. After downloading the archive, it should be unpacked into the `dataset` folder. In case you change the path to the datasets, be sure to specify these changes in the `config/config.py` file by changing the value of the corresponding variable.
### Virtual environment
In the main project directory you will find file `requirements.txt`, which you can use to quickly and easily install the virtual environment on your device with all the necessary libraries.
### Project exploration
The best way to explore my solution of this problem is to study the notebooks - they most clearly demonstrate the stages of solving the problem, the course of my thoughts during this, as well as the main difficulties I encountered.

# Solution
### Data analysis
The data analysis with all explanations and conclusions is presented in the corresponding notebook. In the process of exploring the datasets, I found out in what format they are presented, how to work with them, and I wrote some functions to process the data. I moved them to a separate `modules/prepare.py` file and used them from there when training and testing the model. The code contains a lot of comments, so I think studying this file will not be difficult.
### Model
To solve the problem, as mentioned above, the U-net architecture for a neural network is used. I do not have access to large computing power, so I used a simplified version of the network, a small batch size and a small amount of training data. And even with these underestimated parameters, it required a very large amount of RAM to train all variants of the models. So, our neural network consists of many layers. All convolutional layers of the neural network, except the resulting one, use the ReLu activation function, as well as the he_normal kernel initializer, which samples from the truncated normal distribution. The resulting convolutional layer of the network uses a sigmoidal neuron activation function. When compiling the models, I specified binary crossentropy as the loss function and Adam as the optimizer. Plots of the variation of model accuracy and loss function values as a function of epoch during model training are presented in the notebook `Modeling with Google Colab.ipynb`. In the process of studying model predictions, I found that a significant improvement in model quality could be made if clustering techniques were used to reduce the number of colors in the images in test data. This could be fully exploited if sufficient computational resources were available. You can see visualizations on this topic in the file `Visual model testing.ipynb`.
Overall, my solution does not claim to be the best, as the resulting models are not sufficiently trained because I used about 2.5% of the entire training sample.

# Conclusion
I would like to conclude by saying that this task requires serious computational power to get a good result. I used Google Colab to train my models, and even so, training the models had very serious limitations: the batch size could not exceed 16 units, and the training dataset could not exceed 5000 images. Also, I had to split the test sample into batches of 1000 images each, and predict the results using the model in these small batches. The result with such limitations was understandably weak. However, despite this, I was able to find that using clustering techniques to reduce the number of colors in each image of test dataset has a positive effect on the quality of the resulting predictions. If more serious computational resources were available, it would be possible to create a much better model.

# Resources
- https://muthu.co/reduce-the-number-of-colors-of-an-image-using-k-means-clustering/
- https://www.youtube.com/watch?v=RaswBvMnFxk&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=13
- https://www.tensorflow.org/api_docs/python/tf/keras/layers
- https://www.kaggle.com/competitions/airbus-ship-detection/overview
- https://github.com/bnsreenu/python_for_microscopists/blob/master/076-077-078-Unet_nuclei_tutorial.py