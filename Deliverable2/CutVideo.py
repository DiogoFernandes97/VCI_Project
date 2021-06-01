import cv2 as cv
import numpy as np
import os.path as fl
from matplotlib import pyplot as plt

# Open the video
cap = cv.VideoCapture('Final_Trim.mp4')

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv.CAP_PROP_FPS), cap.get(cv.CAP_PROP_FRAME_COUNT)

# Here you can define your croping values
x, y, w, h = 85, 0, w_frame-85, h_frame

# output  Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('result.avi', fourcc, fps, (w, h))

# Now we start
while cap.isOpened():
    ret, frame = cap.read()

    cnt += 1    # Counting frames

    # Avoid problems when video finish
    if ret:
        # Croping the frame
        crop_frame = frame[y:y+h, x:x+w]

        # Percentage
        xx = cnt * 100 / frames
        print(int(xx), '%')

        # Saving from the desired frames
        # if 15 <= cnt <= 90:
        #    out.write(crop_frame)

        # I see the answer now. Here you save all the video
        out.write(crop_frame)

        # Just to see the video in real time
        cv.imshow('frame', frame)
        cv.imshow('croped', crop_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()