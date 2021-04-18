import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('opencv_logo.png')

# Averaging
blur = cv.blur(img,(5,5))
 
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


# Gaussian Blurring
gaussian_blur = cv.GaussianBlur(img,(5,5),0)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gaussian_blur),plt.title('Gaussian Blurring')
plt.xticks([]), plt.yticks([])
plt.show()

# Median Blurring
median = cv.medianBlur(img,5)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Median Blurring')
plt.xticks([]), plt.yticks([])
plt.show()

# Bilateral Filtering
bilateral_blur = cv.bilateralFilter(img,9,75,75)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(bilateral_blur),plt.title('Bilateral Filtering')
plt.xticks([]), plt.yticks([])
plt.show()