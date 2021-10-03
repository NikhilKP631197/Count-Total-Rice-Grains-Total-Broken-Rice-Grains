#importing all the dependancies
import cv2
import numpy as np
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout

IMG_SIZE = (224,224) #setting the image size for the neural network

train = [] #list to store the training data

DIR = r'E:\Rice Grain Detection\data\train' #address to the directory of training data
category = ['broken_train', 'full_train'] #folder names for the respective classes

#iterating through both the folders and adding the images to the training data list
for c in category:
    folder =  os.path.join(DIR, c)
    label = category.index(c) #0 for broken, 1 for full
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img) 
        img_arr = cv2.imread(img_path) #reading the images
        img_arr = cv2.resize(img_arr, IMG_SIZE) #resizing the images to the size described above for CNN
        train.append([img_arr, label]) #returning the data along with labels
        
#shuffling the data so the model can learn better
random.shuffle(train)

#break the dataset and store the features in X_train and labels in y_train
X_train = []
y_train = []

for features, labels in train:
    X_train.append(features)
    y_train.append(labels)
    
#converting the data into numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

#normalising the pixel values
X_train = X_train/255


#creating the CNN model for classiying
model = Sequential()

#first CNN layer with 32 layers and feature extractor of size 3x3
model.add(Conv2D(32, (3, 3), input_shape = X_train.shape[1:]))
model.add(Activation('relu')) #Rectified Linear Unit as Activation function
model.add(MaxPooling2D(pool_size = (2, 2))) #max pooling layer of size 2x2

#second CNN layer with 32 layers and feature extractor of size 3x3
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu')) #Rectified Linear Unit as Activation function
model.add(MaxPooling2D(pool_size = (2, 2))) #max pooling layer of size 2x2

#third CNN layer with 64 layers and feature extractor of size 3x3
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu')) #Rectified Linear Unit as Activation function
model.add(MaxPooling2D(pool_size = (2, 2))) #max pooling layer of size 2x2

model.add(Flatten()) #Flattening to get 1D array of features
model.add(Dense(64)) #defining the hidden layer with 64 neurons
model.add(Activation('relu')) #Rectified Linear Unit as Activation function
model.add(Dropout(0.5)) 
model.add(Dense(2))
model.add(Activation('softmax')) #activation function that returns the probability of a data lying in either classes

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy']) #compiling the model with the given parameters

model.fit(X_train, y_train, epochs = 10, validation_split = 0.1) #training the model over the training dataset

model.save("grain_classifier.h5") #saving the CNN classifier model