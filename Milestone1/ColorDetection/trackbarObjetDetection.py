import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

cv.namedWindow("Track_Detection")
cv.createTrackbar("L_H", "Track_Detection", 0, 179, nothing)
cv.createTrackbar("L_S", "Track_Detection", 0, 255, nothing)
cv.createTrackbar("L_V", "Track_Detection", 0, 255, nothing)
cv.createTrackbar("U_H", "Track_Detection", 179, 179, nothing)
cv.createTrackbar("U_S", "Track_Detection", 255, 255, nothing)
cv.createTrackbar("U_V", "Track_Detection", 255, 255, nothing)

image = cv.resize(cv.imread("lego.jpg"),(360,360))

while 1:
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # get current positions of trackbars
    l_h = cv.getTrackbarPos("L_H", "Track_Detection")
    l_s = cv.getTrackbarPos("L_S", "Track_Detection")
    l_v = cv.getTrackbarPos("L_V", "Track_Detection")
    u_h = cv.getTrackbarPos("U_H", "Track_Detection")
    u_s = cv.getTrackbarPos("U_S", "Track_Detection")
    u_v = cv.getTrackbarPos("U_V", "Track_Detection")

    # set lower values of HSV
    lower_values = np.array([l_h, l_s, l_v])
    # set upper values of HSV
    upper_values = np.array([u_h, u_s, u_v])
    
    # generate Mask
    mask = cv.inRange(hsv_image, lower_values, upper_values)
    # Result image
    result = cv.bitwise_and(image, image, mask=mask)

    cv.imshow("Original Image", image)
    cv.imshow("Mask", mask)
    cv.imshow("Result", result)

    if cv.waitKey(1) & 0xFF == 27:
        print(lower_values)
        print(upper_values)
        break

cv.destroyAllWindows()