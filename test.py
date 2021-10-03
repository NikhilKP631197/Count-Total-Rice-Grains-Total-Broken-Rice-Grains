#importing all the dependencies
import cv2
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

#importing the output csv file, the background image, and the saved model
op = pd.read_csv('data/submission.csv')
bg = cv2.imread('black.jpg')
model = load_model('grain_classifier.h5')

DIR = r'E:\Rice Grain Detection\data\test' #the address to the tesing data
img = cv2.imread(r'data\test\image_5.jpg') #reading the test images one by one

#converting the test image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11,11), 0) #blurring the images to avoid noise
canny = cv2.Canny(blur, 30, 150, 3) #reading the edges of the grains
dilate = cv2.dilate(canny, (1,1), iterations = 2) #dilating the image to sharpen and thicken the edges

#getting the contours and storring it in a list
(cnt, heirarchy) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#method to scale the contour coordinates to the desired size
def scale_contour(contour, scale):
    moments = cv2.moments(contour) #finding the moments
    midX = 0
    midY = 0
    if (moments['m00']!=0):
        midX = int(round(moments["m10"] / moments["m00"]))
        midY = int(round(moments["m01"] / moments["m00"]))
    mid = np.array([midX, midY])
    contour = contour - mid
    contour = (contour * scale).astype(np.int32) #scaling the contours to a desired size
    contour = contour + mid
    return contour

#painting the contours on to the black background and saving the images in the given directory's address
epoch = 0
for i in cnt:
    bg_copy = cv2.imread('black.jpg') #reading the black image
    cnt_scaled = scale_contour(i, 10) #resizing the contour
    cv2.fillPoly(bg_copy, pts =[cnt_scaled], color=(255,255,255)) #painting the contours on to the black image
    cv2.imwrite(r'data\test\test_image5\image_%04d.jpg'%(epoch+1), bg_copy) #saving the image in the desired address
    epoch+=1
    
op.iloc[4, 1] = int(len(cnt))#the number of contours is the number of grains

folder = r'data\test\test_image5' #folder for the contour images of the test dataset
X_test = []
for img in os.listdir(folder):
    img = os.path.join(folder, img)
    img_arr = cv2.imread(img) #reading the images
    img_arr = cv2.resize(img_arr, (224, 224)) #resizing the images to the desired size for the network to read
    X_test.append(img_arr) #Adding it to the list of test data
    
X_test = np.array(X_test) #converting the list to array as neural network expects numpy as the input array

pred = model.predict(X_test) #predict the probablities of the contours to be of a roken rice grain or a full rice grain
pred = pred.tolist() #converting the prediction array to list for our convinience

count = 0
for [broken, full] in pred:
    if(broken>full):
        count+=1 #counting the number of data that has probability of being broken grain greater than that of being a full grain
        
op.iloc[4,2] = count #the count is the number of broken grains in the image

op.to_csv('data/submission.csv') #adding the values to the op dataframe and saving it to the final csv file