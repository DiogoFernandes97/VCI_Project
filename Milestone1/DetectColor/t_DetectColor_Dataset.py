import cv2 as cv
import numpy as np
import os.path as fl
from matplotlib import pyplot as plt

f = open("Ranges_File.txt", "r")
nonempty_lines = [line.strip("\n") for line in f if line != "\n"]
line_count = len(nonempty_lines)
f.close()

if line_count % 3 != 0: # para garantir que temos todos os valores necessarios
    exit()

print(line_count)

num_range = line_count//3
print(num_range)
#name = np.zeros(num_range, dtype=str_)
name = []
lower_value = np.zeros([num_range, 3])
upper_value = np.zeros([num_range, 3])

f = open("Ranges_File.txt", "r")

image1 = cv.imread('lego_1.jpg')
image2 = cv.imread('lego_2.jpg')
image3 = cv.imread('lego_3.jpg')
image_o = np.concatenate((image1, image2), axis=1)
image_o = np.concatenate((image_o, image3), axis=1)


image = cv.blur(image_o, (50, 50))

mask = np.zeros(image.shape[:2], np.uint8)

hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

for i in range(0, num_range):
    name.append(f.readline().rstrip(":\n"))
    print(name[i])

    x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
    lower_value[i, 0] = x[0]
    lower_value[i, 1] = x[1]
    lower_value[i, 2] = x[2]
    
    x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
    upper_value[i, 0] = x[0]
    upper_value[i, 1] = x[1]
    upper_value[i, 2] = x[2]

    # generate Mask
    mask = mask | cv.inRange(hsv_image, lower_value[i, :], upper_value[i, :])

# Result image
result = cv.bitwise_and(image_o, image_o, mask=mask)

image_or = cv.resize(image_o, (1080, 480))
image_b = cv.resize(image, (1080, 480))
mask_r = cv.resize(mask, (1080, 480))
result_r = cv.resize(result, (1080, 480))


contours, hierarchy = cv.findContours(mask_r, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(result_r, contours, -1, (0, 0, 255), 2)

cv.imshow("Original Image", image_or)
cv.imshow("Blurred", image_b)
cv.imshow("Mask", mask_r)
cv.imshow("Result", result_r)

cv.waitKey(0) & 0xFF == 27

f.close()
cv.destroyAllWindows()