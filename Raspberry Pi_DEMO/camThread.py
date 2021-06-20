# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from gpiozero import LED
import numpy as np
from threading import Thread,Timer,Event
import threading



class camThread:
    def __init__(self, resolution=(800, 608), framerate=40):
        self.wait_event = threading.Event()
        
        # initialize the camera and stream
        self.framerate = framerate
        self.camera = PiCamera()
        #self.camera.sensor_mode = 3
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.awb_mode = 'off'
        self.camera.awb_gains = (1.4375,1.984375)
        #camera.drc_strength = 'high'
        self.camera.iso = 100
        self.camera.exposure_mode = 'off'
        self.camera.shutter_speed = 1800
        
        
        self.image = np.empty((resolution[::-1]+(3,)),dtype = np.uint8)
        self.discard = np.empty((resolution[::-1]+(3,)),dtype = np.uint8)
        self.image2 = np.empty((resolution[::-1]+(3,)),dtype = np.uint8)
        
        
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while  not self.stopped:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
#            self.camera.start_preview()
            if(self.wait_event.isSet()== True):
                self.camera.capture(self.image, format = 'bgr',use_video_port = True)
            # if the thread indicator variable is set, stop the thread
                self.wait_event.clear()
            #else:
                #self.camera.capture(self.discard, format = 'bgr',use_video_port = True)
                # and resource camera resources
        if self.stopped:
            #self.stream.close()
            self.camera.close()
        return
    def read(self):
        # return the frame most recently read
        return self.image
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
