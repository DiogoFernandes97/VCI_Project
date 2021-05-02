import cv2 as cv
import numpy as np
import os.path as fl
from matplotlib import pyplot as plt

def nothing(x):
    pass

def saveData(fileName, low_values, upper_values):
    f = open(fileName, "a")
    rangeColor = str(input("Nome da gama a inserir: "))
    f.write(rangeColor)
    f.write(":\n")
    f.write(np.array2string(lower_values))
    f.write("\n")
    f.write(np.array2string(upper_values))
    f.write("\n")
    f.close()

cv.namedWindow("Track_Detection")
cv.createTrackbar("L_H", "Track_Detection", 0, 179, nothing)
cv.createTrackbar("L_S", "Track_Detection", 0, 255, nothing)
cv.createTrackbar("L_V", "Track_Detection", 0, 255, nothing)
cv.createTrackbar("U_H", "Track_Detection", 179, 179, nothing)
cv.createTrackbar("U_S", "Track_Detection", 255, 255, nothing)
cv.createTrackbar("U_V", "Track_Detection", 255, 255, nothing)

image = cv.imread('lego_2.jpg')

while True:
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

    image_r = cv.resize(image, (360,480))
    mask_r = cv.resize(mask, (360,480))
    result_r = cv.resize(result, (360,480))

    cv.imshow("Original Image", image_r)
    cv.imshow("Mask", mask_r)
    cv.imshow("Result", result_r)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        print(lower_values)
        print(upper_values)
        break  

    if k == ord('s'):
        fileName = str("Ranges_File.txt")
        saveData(fileName, lower_values, upper_values)
    
cv.destroyAllWindows()