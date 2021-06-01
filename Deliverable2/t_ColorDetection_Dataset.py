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

def balance_white(img):
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    bal_image = wb.balanceWhite(img)
    return bal_image

cv.namedWindow("Track_Detection")
cv.createTrackbar("L_H", "Track_Detection", 0, 179, nothing)
cv.createTrackbar("L_S", "Track_Detection", 0, 255, nothing)
cv.createTrackbar("L_V", "Track_Detection", 0, 255, nothing)
cv.createTrackbar("U_H", "Track_Detection", 179, 179, nothing)
cv.createTrackbar("U_S", "Track_Detection", 255, 255, nothing)
cv.createTrackbar("U_V", "Track_Detection", 255, 255, nothing)

image3 = cv.imread('frame.jpg')

image = balance_white(image3)

kernel = np.ones((5,5),np.uint8)

image1 = cv.GaussianBlur(image, (5, 5), 0)

while True:
    hsv_image = cv.cvtColor(image1, cv.COLOR_BGR2HSV)

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
    mask_r1 = cv.inRange(hsv_image, lower_values, upper_values)

    # morphological operation
    #mask_r1 = cv.morphologyEx(mask_r1,cv.MORPH_CLOSE, kernel)
    mask_r1 = cv.morphologyEx(mask_r1, cv.MORPH_OPEN, kernel, iterations=1)
    mask_r1 = cv.dilate(mask_r1, kernel, iterations=1)
    mask_r1 = cv.morphologyEx(mask_r1, cv.MORPH_CLOSE, kernel, iterations=3)
    mask_r1 = cv.erode(mask_r1, kernel, iterations=1)
    #mask_r1 = cv.morphologyEx(mask_r1, cv.MORPH_OPEN, kernel, iterations=6)
    #mask_r1 = cv.dilate(mask_r1, kernel, iterations=10)
    #mask_r1 = cv.erode(mask_r1, kernel, iterations=8)

    # Result image1
    result_r1 = cv.bitwise_and(image1, image1, mask=mask_r1)

    cv.imshow("Original Image", image1)
    cv.imshow("Mask", mask_r1)
    cv.imshow("Result", result_r1)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        print(lower_values)
        print(upper_values)
        break  

    if k == ord('s'):
        fileName = str("Ranges_File.txt")
        saveData(fileName, lower_values, upper_values)
    
cv.destroyAllWindows()