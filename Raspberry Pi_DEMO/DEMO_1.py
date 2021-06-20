# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from gpiozero import LED
backlight = LED(17)
toplight = LED(27)
backlight.off()
toplight.on()
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.sensor_mode = 3
camera.resolution = (800,608) 
camera.framerate = 8
camera.awb_mode = 'off'
camera.awb_gains = (1.4375,1.984375)
#camera.drc_strength = 'high'
camera.iso = 100
camera.exposure_mode = 'off'
camera.shutter_speed = 1800
rawCapture = PiRGBArray(camera, size=(800,608))
# allow the camera to warmup
time.sleep(0.1)

showmask = False

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    #new_frame_time = time.time()
    #image2 = cv2.resize(frame.array,(640,480))
    # show the frame
    cv2.imshow("Frame", image)
    if showmask:
        image2_g = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image2_g = cv2.blur(image2_g,(10,10))
        ret,image2_g = cv2.threshold(image2_g,40,255,cv2.THRESH_BINARY_INV)
        cv2.imshow("Mask",image2_g)
    else:
        try:      
            cv2.destroyWindow("Mask")
        except:
            pass
    rawCapture.truncate(0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
    elif key == ord("b"):
        toplight.toggle()
        backlight.toggle()
    elif key == ord("z"):
        showmask = not showmask

    
camera.close()
toplight.off()
backlight.off()