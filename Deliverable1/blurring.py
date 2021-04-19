import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('quack.jpg')
height, width = img.shape[:2]

#Image Zoom and crop
zoom_amount = 2
x_start=0
y_start=0

width_zoom=width/2
height=height/2

def zoom_crop(img):
    img_zoom = img[x_start:x_start+width,y_start:y_start+width]
    img_zoom = cv.resize(img_zoom,None,fx=zoom_amount, fy=zoom_amount, interpolation = cv.INTER_CUBIC)
    return img_zoom

img_zoom = zoom_crop(img)

# Averaging
blur = cv.blur(img,(5,5))
blur_zoom = zoom_crop(blur)

image1 = np.hstack((img_zoom, blur_zoom))

cv.imshow('Averaging', image1)
cv.waitKey(0)

# Gaussian Blurring
gaussian_blur = cv.GaussianBlur(img,(5,5),0)

gaussian_blur_zoom = zoom_crop(gaussian_blur)

image2 = np.hstack((img_zoom, gaussian_blur_zoom))
cv.imshow('Gaussian Blurring', image2)
cv.waitKey(0)

# Median Blurring
median = cv.medianBlur(img,5)

median_zoom = zoom_crop(median)

image3 = np.hstack((img_zoom, median_zoom))
cv.imshow('Median Blurring', image3)
cv.waitKey(0)

# Bilateral Filtering
bilateral_blur = cv.bilateralFilter(img,9,75,75)

bilateral_blur_zoom = zoom_crop(bilateral_blur)

image4 = np.hstack((img_zoom, bilateral_blur_zoom))
cv.imshow('Bilateral Blurring', image4)
cv.waitKey(0)

cv.destroyAllWindows()
