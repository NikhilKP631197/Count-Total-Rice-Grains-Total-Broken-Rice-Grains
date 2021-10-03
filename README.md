# Count Total Rice Grains & Total Broken Rice Grains
Using OpenCV to calculate the total number of rice grains in a picture and the training a CNN model to distinguish between a broken rice grain and full rice grain and finally count the number of each.

## Modules and Frameworks Used:

1.) OpenCV
2.) Numpy
3.) Pandas
4.) Tensorflow/keras
5.) os
6.) Random
7.) Matplotlib

## File Description:

1.) Data
    a.) test
        i.)   test_image1 - contours of test image 1
        ii.)  test_image2 - contours of test image 2
        iii.) test_image3 - contours of test image 3
        iv.)  test_image4 - contours of test image 4
        v.)   test_image5 - contours of test image 5
        vi.)  image_1.jpg to image_5.jpg - test images
    b.) train
        i.)   broken - contains the broken grain images
        ii.)  broken_train - contains the contour of each broken grain from all the images
        iii.) full - contains the full grain images
        iv.)  full_train - contains the contour of each full grain from all the images
        v.)   mixed_grain_1.jpg to mixed_grain_2.jpg - training images containing both the broken and full grain of rices
    c.) submission.csv - stores the total number of grains and total number of broken grains each for the respective images
2.) black.jpg - A total black image used as background for painting the scaled contours of rice grains onto it
3.) create_training_set.py - Python source code to preprocess the training data, finding the contours, painting contours of each rice grain and saving the corresponding images in the given directory
4.) grain_classifier.h5 - trained CNN model to classify the given contour images as either broke or full grain of rice
5.) model.PNG - screenshot of the loss and accuracy of the model at each epoch
6.) model.txt - text file for the loss and accuracy of the model at each epoch
7.) test.py - Python source code to import the test images, preprocess them, paint the contours of each grain, saving them, and importing them into array to make predictions and couting the total rice grains and total broken rice grains
8.) train_model.py - Python source code to train the cnn model to classify broken grains and full grains and save the model
