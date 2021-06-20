import time
from gpiozero import LED
import numpy as np
from threading import Thread,Timer,Event
import threading



class lightsThread:
    def __init__(self):
        self.backlight = LED(17)
        self.toplight = LED(27)
        self.backlight.on()
        self.toplight.off()
        
        
        self.stopped = False
        self.wait_event = threading.Event()
    
    def start(self,cam):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=(cam,)).start()
        return self
    def update(self,cam):
        # keep looping infinitely until the thread is stopped
        while  not self.stopped:
            
            self.wait_event.wait()
            self.backlight.toggle()
            self.toplight.toggle()
            time.sleep(0.01)
            self.wait_event.clear()
            cam.wait_event.set()
            
        if self.stopped:
            self.backlight.off()
            self.toplight.off()
        return
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.backlight.off()
        self.toplight.off()
