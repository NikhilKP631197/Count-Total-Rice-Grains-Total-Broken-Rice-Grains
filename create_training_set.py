#importing all the dependencies
import cv2
import numpy as np
import os

#importing black image as background to display the contour of each grain
bg = cv2.imread("black.jpg")
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB) 

#address to the directories of broken rice grain training set and full rice grain training set
DIR_broken = r'E:\Rice Grain Detection\data\train\broken'
DIR_full = r'E:\Rice Grain Detection\data\train\full'

#storing the shape of background image that is to be used to resize the training images to fit the background
IMG_SIZE = (bg.shape[1], bg.shape[0])

#creating lists to store the training images of respective classes
broken_data = []
full_data = []

#importing all the images of broken grains, resizing them to the size of background image
for img in os.listdir(DIR_broken):
    img_path = os.path.join(DIR_broken, img)
    img_arr = cv2.imread(img_path)
    img_arr = cv2.resize(img_arr, IMG_SIZE)
    broken_data.append(img_arr)

#importing all the images of broken grains, resizing them to the size of background image
for img in os.listdir(DIR_full):
    img_path = os.path.join(DIR_full, img)
    img_arr = cv2.imread(img_path)
    img_arr = cv2.resize(img_arr, IMG_SIZE)
    full_data.append(img_arr)
    
#method to preprocess the images
def img_preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting the image to grayscale
    blur = cv2.GaussianBlur(gray, (11,11), 0) #blurring the image to avoid noise
    canny = cv2.Canny(blur, 30, 150, 3) #getting the edges
    dilate = cv2.dilate(canny, (1,1), iterations = 2) #sharpening the edges
    return dilate

#lists to append the processed images
broken = []
full = []

for i in broken_data:
    broken.append(img_preprocess(i))

for i in full_data:
    full.append(img_preprocess(i))
    
#finding the contours of the broken rice grains and storing them in a list
broken_cnt = []
for i in broken:
    (cnt, h_broken) = cv2.findContours(i.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    broken_cnt.append(cnt)
    
#finding the contours of the full rice grains and storing them in a list
full_cnt = []
for i in full:
    (cnt, f_broken) = cv2.findContours(i.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    full_cnt.append(cnt)
    
#method to increase the size of contours to print on the background image
def scale_contour(contour, scale):
    moments = cv2.moments(contour) #finding the moments of the contour
    midX = 0
    midY = 0
    if (moments['m00']!=0):
        midX = int(round(moments["m10"] / moments["m00"]))
        midY = int(round(moments["m01"] / moments["m00"]))
    mid = np.array([midX, midY])
    contour = contour - mid
    contour = (contour * scale).astype(np.int32) #scaling the contour coordinates to the desired size
    contour = contour + mid
    return contour

epoch = 0
#putting the contours of the broken rice grain filled in white on the black background and saving it in the given address
for j in broken_cnt:
    for i in range(0, len(j)):
        bg_copy = bg.copy()
        cnt_scaled = scale_contour(j[i], 10)
        cv2.fillPoly(bg_copy, pts =[cnt_scaled], color=(255,255,255)) #filling in the contours with white
        cv2.imwrite(r'data\train\broken_train\broken_%04d.png'%(epoch+1), bg_copy)
        epoch+=1

#putting the contours of the full rice grain filled in white on the black background and saving it in the given address       
epoch = 0
for j in full_cnt:
    for i in range(0, len(j)):
        bg_copy = bg.copy()
        cnt_scaled = scale_contour(j[i], 10)
        cv2.fillPoly(bg_copy, pts =[cnt_scaled], color=(255,255,255))
        cv2.imwrite(r'data\train\full_train\full_%04d.png'%(epoch+1), bg_copy) #filling in the contours with white
        epoch+=1