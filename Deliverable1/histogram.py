import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('wiki.jpg', 0)
img_equalized = cv.equalizeHist(img)

hist = cv.calcHist([img],[0],None,[256],[0,256])
hist_equalized = cv.calcHist([img_equalized],[0],None,[256],[0,256])

plt.subplot(121), plt.title('Histogram'), plt.plot(hist)
plt.subplot(122), plt.title('Equalized Histogram'), plt.plot(hist_equalized)

res = np.hstack((img,img_equalized)) #stacking images side-by-side
cv.imshow('Result',res)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()