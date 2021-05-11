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

if line_count % 3 != 0: # para garantir que temos todos os valores necessarios
    exit()

print(line_count)

num_range = line_count//3
print(num_range)
#name = np.zeros(num_range, dtype=str_)
name = []
lower_value = np.zeros([num_range, 3])
upper_value = np.zeros([num_range, 3])

f = open("Ranges_File_1.txt", "r")

image = cv.imread('lego_3.jpg')
image_b = cv.blur(image, (50, 50))
image_b=cv.Canny(image_b,10,50) #apply canny to roi
mask = np.zeros(image_b.shape[:2], np.uint8)

# kernel for morphological
kernel = np.ones((5,5),np.uint8)

# balance white
image = balance_white(image)

hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

for i in range(0, num_range):
    name.append(f.readline().rstrip(":\n"))
    #print(name[i])

    x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
    lower_value[i,0] = x[0]
    lower_value[i,1] = x[1]
    lower_value[i,2] = x[2]
    
    x = ((f.readline().lstrip("[")).rstrip("]\n")).split()
    upper_value[i,0] = x[0]
    upper_value[i,1] = x[1]
    upper_value[i,2] = x[2]   

    
    # generate Mask
    mask = mask | cv.inRange(hsv_image, lower_value[i,:], upper_value[i,:])
    
    erosion = cv.erode(mask,kernel,iterations = 1)
    #mask = cv.morphologyEx(image_b, cv.MORPH_CLOSE, kernel)
    mask = cv.dilate(erosion,kernel,iterations = 2)
    #mask = cv.morphologyEx(mask, cv.MORPH_BLACKHAT, kernel)
# Result image
result = cv.bitwise_and(image, image, mask=mask)

image_r = cv.resize(image, (360,480))
mask_r = cv.resize(mask, (360,480))
result_r = cv.resize(result, (360,480))

contours, hierarchy = cv.findContours(mask_r, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(result_r, contours, -1, (0, 0, 255), 2)

cv.imshow("Original Image", image_r)
cv.imshow("Mask", mask_r)
cv.imshow("Result", result_r)

cv.waitKey(0) & 0xFF == 27

f. close()
cv.destroyAllWindows()