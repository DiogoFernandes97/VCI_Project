import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def CLAHE_equalize_hist_color_hsv(img2equalize):
    """
    Equalize the image splitting the image after HSV conversion and applying cv.equalizeHist()
    to the V channel, merging the channels and convert back to the BGR color space
    """
    H, S, V = cv.split(cv.cvtColor(img2equalize, cv.COLOR_BGR2HSV))
    eq_V = cv.equalizeHist(V)
    eq_V = cv.equalizeHist(eq_V)
    # CLAHE
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq_V = clahe.apply(eq_V)
    eq_image = cv.cvtColor(cv.merge([H, S, eq_V]), cv.COLOR_HSV2BGR)
    return eq_image
    
'''
def equalize_hist_color_hsv(img):
    H, S, V = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    eq_V = cv.equalizeHist(V)
    eq_image = cv.cvtColor(cv.merge([H, S, eq_V]), cv.COLOR_HSV2BGR)
    return eq_image
'''

def balance_white(img):
    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    bal_image = wb.balanceWhite(img)
    return bal_image

image = cv.imread('lego_light.jpg')

img_balanced = balance_white(image)
img_equalized = CLAHE_equalize_hist_color_hsv(img_balanced)
# img_equalized = equalize_hist_color_hsv(img_balanced)

img_r = cv.resize(image,(360,480))
img_balanced_r = cv.resize(img_balanced,(360,480))
img_equalized_r = cv.resize(img_equalized,(360,480))

cv.imshow('Original_Image',img_r)
cv.imshow('Balanced_Image',img_balanced_r)
cv.imshow('Equalized_Image',img_equalized_r)

cv.imshow('Original_Image',img_r)
cv.imshow('Equalized_Image',img_equalized_r)

cv.waitKey(0)

cv.destroyAllWindows()