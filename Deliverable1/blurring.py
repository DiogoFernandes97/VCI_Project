import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('parrots.jpg')

# Averaging
blur = cv.blur(img,(5,5))

image1 = np.hstack((img, blur))
cv.imshow('Averaging', image1)
cv.waitKey(0)

# Gaussian Blurring
gaussian_blur = cv.GaussianBlur(img,(5,5),0)

image2 = np.hstack((img, gaussian_blur))
cv.imshow('Gaussian Blurring', image2)
cv.waitKey(0)

# Median Blurring
median = cv.medianBlur(img,5)

image3 = np.hstack((img, median))
cv.imshow('Median Blurring', image3)
cv.waitKey(0)

# Bilateral Filtering
bilateral_blur = cv.bilateralFilter(img,9,75,75)

image4 = np.hstack((img, bilateral_blur))
cv.imshow('Bilateral Blurring', image4)
cv.waitKey(0)

cv.destroyAllWindows()