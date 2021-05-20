import cv2 as cv
import numpy as np
import os.path as fl
from matplotlib import pyplot as plt
from numpy.lib.type_check import imag

def balance_white(img):
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    bal_image = wb.balanceWhite(img)
    return bal_image

# variable initialization
name = []
unit_size = 162.0

#c_area = []

f = open("Ranges_File.txt", "r")
nonempty_lines = [line.strip("\n") for line in f if line != "\n"]
line_count = len(nonempty_lines)
f.close()

if line_count % 3 != 0:  # para garantir que temos todos os valores necessarios
    exit()

num_range = line_count // 3
print(num_range)

lower_value = np.zeros([num_range, 3])
upper_value = np.zeros([num_range, 3])

f = open("Ranges_File.txt", "r")

image = cv.imread('lego_3.jpg')
image_b = cv.blur(image, (50, 50))
image_b = cv.Canny(image_b, 10, 50)  # apply canny to roi
mask = np.zeros(image_b.shape[:2], np.uint8)

# kernel for morphological
kernel = np.ones((5, 5), np.uint8)

# balance white
image = balance_white(image)

hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

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

    mask_t = cv.inRange(hsv_image, lower_value[i, :], upper_value[i, :])

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

contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)   # tentar cv.RETR_EXTERNAL
coordinates = np.zeros([len(contours), 3])

i=0
for c in contours:
    
    # compute the center of the contour
    M = cv.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print('Pos_x:',cX,' Pos_y:',cY)
  
    coordinates[i,0]=cX # coordenada x
    coordinates[i,1]=cY # coordenada y

    # draw the contour and center of the pieces on the image
    #cv.drawContours(result, [c], -1, (0, 0, 255), 8)
    cv.circle(result, (cX, cY), 20, (255, 100, 100), -1) 

    # Identify the minimun area of the lego piece
    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    result = cv.drawContours(result,[box],0,(0,255,0),10)
    ((x, y), (width, height), angle)=cv.minAreaRect(c)

    print('width:', round(width/unit_size), '   ', 'height', round(height/unit_size))
    cv.putText(result, str(i)+" "+ str(int(round(width/162,0)))+"x"+str(int(round(height/162,0))  ), (cX - 200, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)
    
    i=i+1


image_r = cv.resize(image, (360, 480))
mask_r = cv.resize(mask, (360, 480))
result_r = cv.resize(result, (360, 480))


cv.imshow("Original Image", image_r)
cv.imshow("Mask", mask_r)
cv.imshow("Result", result_r)

cv.waitKey(0) & 0xFF == 27

f.close()
cv.destroyAllWindows()


# To do
#   - tamanho em cm;
#   - area
#   - perimetro