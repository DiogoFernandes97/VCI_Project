import cv2
import numpy as np

def nothing(x):
    pass

# Load image
img1 = cv2.imread('dataset/01/lego-rot0-1a.jpg')
img2 = cv2.imread('dataset/01/lego-rot0-1c.jpg')
img3 = cv2.imread('dataset/01/lego-rot0-2a.jpg')
img4 = cv2.imread('dataset/01/lego-rot0-3c.jpg')
# Resize to fit screen (lose definition, only for visual while coding)
image1 = cv2.resize(img1, (300, 300))
image2 = cv2.resize(img2, (300, 300))
image3 = cv2.resize(img3, (300, 300))
image4 = cv2.resize(img4, (300, 300))

numpy_vertical = np.vstack((image1, image2))
numpy_vertical2 = np.vstack((image3, image4))
numpy_horizontal = np.hstack((numpy_vertical, numpy_vertical2))
image1 = np.concatenate((numpy_vertical, numpy_vertical2), axis=1)

#image = np.concatenate((image1, image2), axis=0)
#image = np.concatenate((image, image3), axis=1)
#image = np.concatenate((image, ))


# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image1, image1, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    cv2.imshow('result', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()