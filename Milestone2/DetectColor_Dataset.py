import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from decimal import Decimal as dec
import math
import LegoClass as lego
import ColorClass as color

def balance_white(img):
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    bal_image = wb.balanceWhite(img)
    return bal_image


# variable initialization
name = []
mm_px = 0.04822

colors = color.Range()
x = colors.parseFile("Ranges_File.txt")
print(colors.num_range)

#f = open("Ranges_File.txt", "r")

image = cv.imread('lego_3.jpg')
image_b = cv.blur(image, (50, 50))
image_b = cv.Canny(image_b, 10, 50)  # apply canny to roi
mask = np.zeros(image_b.shape[:2], np.uint8)
lego_mask = np.zeros(image_b.shape[:2], np.uint8)

# kernel for morphological
kernel = np.ones((5, 5), np.uint8)

# balance white
image = balance_white(image)

hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

for i in range(0, colors.num_range):

    mask_t = cv.inRange(hsv_image, colors.lower_value[i, :], colors.upper_value[i, :])

    mask_t = cv.morphologyEx(mask_t, cv.MORPH_OPEN, kernel, iterations=2)
    mask_t = cv.dilate(mask_t, kernel, iterations=2)
    mask_t = cv.morphologyEx(mask_t, cv.MORPH_CLOSE, kernel, iterations=10)
    mask_t = cv.erode(mask_t, kernel, iterations=1)
    mask_t = cv.morphologyEx(mask_t, cv.MORPH_OPEN, kernel, iterations=6)
    mask_t = cv.dilate(mask_t, kernel, iterations=10)
    mask_t = cv.erode(mask_t, kernel, iterations=8)

    # generate Mask
    mask = mask | mask_t

# Result image
result = cv.bitwise_and(image, image, mask=mask)



contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)   # tentar cv.RETR_EXTERNAL
coordinates = np.zeros([len(contours), 3])

Pieces = []

i = 0
for c in contours:
    
    # compute the center of the contour
    M = cv.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
  
    coordinates[i, 0] = cX  # coordenada x
    coordinates[i, 1] = cY  # coordenada y

    # draw the contour and center of the pieces on the image
    #cv.drawContours(result, [c], -1, (0, 0, 255), 8)
    #cv.circle(result, (cX, cY), 20, (255, 100, 100), -1) 

    # Identify the minimun area of the lego piece
    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    result = cv.drawContours(result, [box], 0, (0, 255, 0), 10)
    ((x, y), (width, height), angle) = cv.minAreaRect(c)

    # calculo das dimensoes em mm
    w_mm = (width *mm_px)
    h_mm = (height*mm_px)

    # converte para nº blocos
    w = lego.mmToBlocks(w_mm)
    h = lego.mmToBlocks(h_mm)

    # area
    area = cv.contourArea(box)
    # perimeter
    perimeter = cv.arcLength(box,True)

    lego_mask = np.zeros(image.shape[:2], np.uint8)
    cv.drawContours(lego_mask,[box],-1,(255),-1)
    mean_val = cv.mean(hsv_image,mask = lego_mask)
    color_index = colors.checkInRange(mean_val[0:3])

    Pieces.append(lego.Piece(w,h,color_index,cX,cY))

    print('Piece',i,':')
    print("Colour: " + colors.getName(color_index) )
    print('width(px):', round(width,1), '   ', 'height(px):', round(height,1))

    print('width(mm):', round(w_mm,1), '   ', 'height(mm):', round(h_mm))
    print('width(blocks):', w, '   ', 'height(blocks):',h)

    print('Area(mm^2):',round(area*math.pow(mm_px,2)))

    print('Perimeter(mm):',round(perimeter*mm_px))
    cv.putText(result, str(i)+" "+ Pieces[i].dimsStr(), (cX - 200, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)
    cv.putText(result, colors.getName(color_index) , (cX -150, cY + 100), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10)
    print('\n')

    i=i+1


image_r = cv.resize(image, (360, 480))
mask_r = cv.resize(mask, (360, 480))
result_r = cv.resize(result, (360, 480))


cv.imshow("Original Image", image_r)
cv.imshow("Mask", mask_r)
cv.imshow("Result", result_r)

cv.waitKey(0) 

cv.destroyAllWindows()
