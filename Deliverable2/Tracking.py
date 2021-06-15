import cv2 as cv
import numpy as np
import os.path as fl
from matplotlib import pyplot as plt
import imutils
from collections import deque
from numpy.core.fromnumeric import reshape, resize, var

from numpy.core.numeric import rollaxis


def balance_white(img):
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    bal_image = wb.balanceWhite(img)
    return bal_image


# Open the video
cap = cv.VideoCapture('Final.mp4')

mm_px = 0.27
# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv.CAP_PROP_FPS), cap.get(cv.CAP_PROP_FRAME_COUNT)


f = open("Ranges_File_5.txt", "r")
nonempty_lines = [line.strip("\n") for line in f if line != "\n"]
line_count = len(nonempty_lines)
f.close()

if line_count % 3 != 0:  # para garantir que temos todos os valores necessarios
    exit()

num_range = line_count // 3
# print(num_range)

lower_value = np.zeros([num_range, 3])
upper_value = np.zeros([num_range, 3])

# kernel for morphological
kernel = np.ones((5, 5), np.uint8)

f = open("Ranges_File_5.txt", "r")
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

x1=-1
# loop over the frames of the video
while cap.isOpened():

    ret, frame = cap.read()

    cnt += 1  # Counting frames

    if ret:
        # Percentage
        xx = int(cnt * 100 / frames)
        if (xx != x1) and (xx%10 == 0):
            x1=xx
            print('Procesing...     ',xx, '%')

        result = frame.copy()
        result = cv.resize(result, (round(w_frame/3), round(h_frame/3)))
        frame2 = frame.copy()
        frame2 = cv.resize(frame2, (round(w_frame/3), round(h_frame/3)))
        frame2 = balance_white(frame2)

        #print('Frame:', cnt)
        '''if cnt == 600:
            cv.imwrite('adjf1.jpg', frame2)''' # get a frame for range calibration
        mask = np.zeros(frame2.shape[:2], np.uint8)
        blurred = cv.GaussianBlur(frame2, (5, 5), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        # construct a mask for colors, then perform
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
            # print(M)
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

            ((x, y), (width, height), angle) = cv.minAreaRect(c)

            # calculo das dimensoes em mm
            w_mm = (width * mm_px)
            h_mm = (height * mm_px)

            # converte para nº blocos
            w = round((w_mm + 0.2) / 8)
            h = round((h_mm + 0.2) / 8)

            # Ensuring even sizes
            if w > 3:
                w = round(w / 2) * 2
            if h > 3:
                h = round(h / 2) * 2

            # area
            # area_1 = w_mm*h_mm
            area = cv.contourArea(box)
            # perimeter
            # perimeter_1 = 2*w_mm + 2*h_mm
            perimeter = cv.arcLength(box, True)

            lego_mask = np.zeros(frame2.shape[:2], np.uint8)
            cv.drawContours(lego_mask, [box], -1, (255), -1)

            mean_val = cv.mean(hsv, mask=lego_mask)

            color_index = 0
            for k in range(0, num_range):
                if np.all(cv.inRange(mean_val[0:3], lower_value[k, :], upper_value[k, :]) == 255):
                    # print("Found Colour " + name[k])
                    color_index = k
                    break

            area_A = cv.contourArea(c)
            if area_A > 800:
                i = i + 1
                result = cv.drawContours(result, [box], 0, (0, 0, 255), 3)
                #result = cv.drawContours(result, contours, -1, (255, 0, 0), 3)
                cv.putText(result, str(i) + " " + str(w) + "x" + str(h), (cX - 40, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 255, 255), 2)
                cv.putText(result, name[color_index], (cX - 40, cY + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                           2)
        # Show
        cv.imshow('Mask', mask)
        cv.imshow('frame2', frame2)
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
