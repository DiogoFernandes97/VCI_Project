import cv2 as cv
import numpy as np
import os.path as fl
from matplotlib import pyplot as plt
import imutils
from collections import deque

def balance_white(img):
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    bal_image = wb.balanceWhite(img)
    return bal_image



# Open the video
cap = cv.VideoCapture('result.avi')

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv.CAP_PROP_FPS), cap.get(cv.CAP_PROP_FRAME_COUNT)

greenLower = (29, 86, 6)
greenUpper = (84, 255, 255)

f = open("Ranges_File.txt", "r")
nonempty_lines = [line.strip("\n") for line in f if line != "\n"]
line_count = len(nonempty_lines)
f.close()

if line_count % 3 != 0:  # para garantir que temos todos os valores necessarios
    exit()

num_range = line_count // 3
#print(num_range)

lower_value = np.zeros([num_range, 3])
upper_value = np.zeros([num_range, 3])



# kernel for morphological
kernel = np.ones((5, 5), np.uint8)

f = open("Ranges_File.txt", "r")
name = []
for i in range(0, num_range):
    name.append(f.readline().rstrip(":\n"))

    x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
    lower_value[i, 0] = x[0]
    lower_value[i, 1] = x[1]
    lower_value[i, 2] = x[2]

    x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
    upper_value[i, 0] = x[0]
    upper_value[i, 1] = x[1]
    upper_value[i, 2] = x[2]
f.close()

# loop over the frames of the video
while cap.isOpened():

    ret, frame = cap.read()

    cnt += 1  # Counting frames

    if ret:
        # Percentage
        xx = cnt * 100 / frames
        print(int(xx), '%')

        result = frame.copy()
        frame2 = frame.copy()
        frame2 = balance_white(frame2)
        print('Frame:', cnt)
        if cnt == 1888:
            cv.imwrite('frame.jpg', frame)

        mask = np.zeros(frame2.shape[:2], np.uint8)

        blurred = cv.GaussianBlur(frame2, (5, 5), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask

        for i in range(0, num_range):
            mask_t = cv.inRange(hsv, lower_value[i, :], upper_value[i, :])

            mask_t = cv.morphologyEx(mask_t, cv.MORPH_OPEN, kernel, iterations=1)
            mask_t = cv.dilate(mask_t, kernel, iterations=1)
            mask_t = cv.morphologyEx(mask_t, cv.MORPH_CLOSE, kernel, iterations=3)
            mask_t = cv.erode(mask_t, kernel, iterations=1)
            mask_t = cv.morphologyEx(mask_t, cv.MORPH_CLOSE, kernel, iterations=3)
            mask_t = cv.dilate(mask_t, kernel, iterations=1)
            mask_t = cv.erode(mask_t, kernel, iterations=1)

            # generate Mask
            mask = mask | mask_t

        i = 0

        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # tentar cv.RETR_EXTERNAL
        coordinates = np.zeros([len(contours), 3])

        for c in contours:
            # compute the center of the contour
            M = cv.moments(c)
            #print(M)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # set values as what you need in the situation
                cX, cY = 0, 0



            coordinates[i, 0] = cX  # coordenada x
            coordinates[i, 1] = cY  # coordenada y

            # Identify the minimun area of the lego piece
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            area_A = cv.contourArea(c)
            if area_A > 800:
                result = cv.drawContours(result, [box], 0, (0, 255, 0), 3)
            ((x, y), (width, height), angle) = cv.minAreaRect(c)






        #Show
        cv.imshow('Mask', mask)
        cv.imshow('frame2', frame2)

        f.close()
        cv.imshow('Show', result)

        # Press Q on keyboard to  exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# When everything done, release the video capture object
cap.release()
# Closes all
cv.destroyAllWindows()



