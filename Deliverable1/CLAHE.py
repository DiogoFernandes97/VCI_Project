import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('tsukuba_l.png',0)
img_equalized = cv.equalizeHist(img)

hist = cv.calcHist([img],[0],None,[256],[0,256])
hist_equalized = cv.calcHist([img_equalized],[0],None,[256],[0,256])

plt.subplot(221,), plt.title('Original Image'), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.subplot(222,), plt.title('Image After Equalization'), plt.imshow(img_equalized, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.title('Histogram'), plt.plot(hist)
plt.subplot(224), plt.title('Equalized Histogram'), plt.plot(hist_equalized)
plt.show()

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

plt.subplot(121), plt.title('Image After Equalization'), plt.imshow(img_equalized, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.title('CLAHE'), plt.imshow(cl1, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()