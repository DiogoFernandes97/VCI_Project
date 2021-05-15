import cv2 as cv
import numpy as np
import os.path as fl
from matplotlib import pyplot as plt


def balance_white(img):
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    bal_image = wb.balanceWhite(img)
    return bal_image


f = open("Ranges_File.txt", "r")
nonempty_lines = [line.strip("\n") for line in f if line != "\n"]
line_count = len(nonempty_lines)
f.close()

if line_count % 3 != 0:  # para garantir que temos todos os valores necessarios
    exit()

print(line_count)

num_range = line_count // 3
print(num_range)
# name = np.zeros(num_range, dtype=str_)
name = []
c_area = []
color_mean = []
lower_value = np.zeros([num_range, 3])
upper_value = np.zeros([num_range, 3])

f = open("Ranges_File.txt", "r")

image = cv.imread('lego_3.jpg')
image_b = cv.blur(image, (50, 50))
image_b = cv.Canny(image_b, 10, 50)  # apply canny to roi
mask = np.zeros(image.shape[:2], np.uint8)

# kernel for morphological
kernel = np.ones((5, 5), np.uint8)

# balance white
image = balance_white(image)

hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

for i in range(0, num_range):
    name.append(f.readline().rstrip(":\n"))
    # print(name[i])

    x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
    lower_value[i, 0] = x[0]
    lower_value[i, 1] = x[1]
    lower_value[i, 2] = x[2]

    x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
    upper_value[i, 0] = x[0]
    upper_value[i, 1] = x[1]
    upper_value[i, 2] = x[2]

    mask_t = cv.inRange(hsv_image, lower_value[i, :], upper_value[i, :])

    color_mean.append((cv.mean(image, mask=mask_t)[:3]))

    mask_t = cv.morphologyEx(mask_t, cv.MORPH_OPEN, kernel, iterations=2)
    mask_t = cv.dilate(mask_t, kernel, iterations=2)
    mask_t = cv.morphologyEx(mask_t, cv.MORPH_CLOSE, kernel, iterations=10)
    mask_t = cv.erode(mask_t, kernel, iterations=1)
    mask_t = cv.morphologyEx(mask_t, cv.MORPH_CLOSE, kernel, iterations=1)
    mask_t = cv.morphologyEx(mask_t, cv.MORPH_OPEN, kernel, iterations=6)
    mask_t = cv.dilate(mask_t, kernel, iterations=10)
    mask_t = cv.erode(mask_t, kernel, iterations=8)

    # generate Mask
    mask = mask | mask_t




# Result image
result = cv.bitwise_and(image, image, mask=mask)

#
color_mean = np.array(color_mean)
color_mean = color_mean.astype(int)

contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for c in contours:
    # compute the center of the contour
    M = cv.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the contour and center of the shape on the image
    cv.drawContours(result, [c], -1, (0, 0, 255), 8)
    cv.circle(result, (cX, cY), 20, (255, 100, 100), -1)
    c_area.append(int(cv.contourArea(c)))
    cv.putText(result, str(int(cv.contourArea(c))), (cX - 200, cY - 20),
        cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)

c_area = np.array(c_area)

image_r = cv.resize(image, (360, 480))
mask_r = cv.resize(mask, (360, 480))
result_r = cv.resize(result, (360, 480))


cv.imshow("Original Image", image_r)
cv.imshow("Mask", mask_r)
cv.imshow("Result", result_r)
print(c_area)
print(color_mean)

cv.waitKey(0) & 0xFF == 27

f.close()
cv.destroyAllWindows()