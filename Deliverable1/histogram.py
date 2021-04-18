import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('wiki.jpg', 0)
img_equalized = cv.equalizeHist(img)

hist = cv.calcHist([img],[0],None,[256],[0,256])
hist_equalized = cv.calcHist([img_equalized],[0],None,[256],[0,256])

plt.subplot(221), plt.title('Original Image'), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.title('Equalized Image'), plt.imshow(img_equalized, cmap='gray', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.title('Histogram'), plt.plot(hist)
plt.subplot(224), plt.title('Equalized Histogram'), plt.plot(hist_equalized)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()