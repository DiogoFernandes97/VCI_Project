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

bgr = cv.resize(cv.imread("lego-rot45-5b_greened.jpg"), (360,480))


lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)

lab_planes = cv.split(lab)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#lab_planes[0] = cv.equalizeHist(lab_planes[0])

lab_planes[0] = clahe.apply(lab_planes[0])

lab = cv.merge(lab_planes)

image = cv.cvtColor(lab, cv.COLOR_LAB2BGR)


wb = cv.xphoto.createGrayworldWB()
wb.setSaturationThreshold(0.99)
image = wb.balanceWhite(image)


while 1:
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    hsv_image[:, :, 2] = cv.equalizeHist(hsv_image[:, :, 2])
    

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