import cv2
import numpy as np
from camThread import camThread
from lightsThread import lightsThread
import ColorClass as color

while_var = True

def wait_func(delay):
    global while_var
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        while_var = False
    
colors = color.Range()
x = colors.parseFile("Ranges_File.txt")
print(colors.num_range)


mm_px = 0.1429

width = 800
height = 608
image = np.empty((608,800,3),dtype = np.uint8)
hsv_image = np.empty((608,800,3),dtype = np.uint8)
image2 = np.empty((608,800,3),dtype = np.uint8)
image2_g = np.empty((608,800,1),dtype = np.uint8)
image_seg = np.empty((608,800,3),dtype = np.uint8)

mask = np.ones((608,800,1),dtype = np.uint8)

cam = camThread().start()
lights = lightsThread().start(cam)

while while_var:
    
    lights.wait_event.set()
   
    wait_func(150)
    
    image = cv2.copyTo(cam.read(),mask)
    cv2.imshow("even", image)
    
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
   
  
    
    lights.wait_event.set()
    
    wait_func(150)
     
    image2 = cv2.copyTo(cam.read(),mask)
    
    image2_g = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    image2_g = cv2.blur(image2_g,(10,10))
    ret,image2_g = cv2.threshold(image2_g,40,255,cv2.THRESH_BINARY_INV)
    image2_g[:,0:40] = 0; #Manter apenas ROI
    
    
    cv2.imshow("odd", image2)
    
    image_seg = cv2.bitwise_and(image,image, mask = image2_g)
    
    #cv2.imshow("segmented",image_seg)

    contours,hierarchy = cv2.findContours(image2_g,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        #cv2.drawContours(image_seg, [c], -1, (0,0,255),8)
        M = cv2.moments(c)
        
        if int(M["m00"]) != 0:
            cX = int(M["m10"]/ M["m00"])
            cY = int(M["m01"]/ M["m00"])
        else:
            cX = 0;
            cY = 0;
            
        rect = cv2.minAreaRect(c)
        ((x,y),(r_width,r_height),angle) = rect;
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        w_mm = (r_width*mm_px)
        h_mm = (r_height*mm_px)
        
        w = round((w_mm + 0.2)/8)
        h = round((h_mm + 0.2)/8)
        
        if w > 3:
            w = 2*round(w/2)
        if h > 3:
            h = 2*round(h/2)
        area =r_width*r_height
        
        if area > 4000:     
            if ((cX > 50) and (cX < width-50)) and ((cY > 40) and (cY < height - 40)): 
                mean_val = 0
                lego_mask = np.zeros(image.shape[:2],np.uint8)
                cv2.drawContours(lego_mask,[box],-1,(255),-1)
                mean_val = cv2.mean(image,mask = lego_mask)
                cv2.drawContours(image_seg,[box],-1,(0,255,0),5)
                mean_val = cv2.mean(hsv_image,mask = lego_mask)
                color_index = colors.checkInRange(mean_val[0:3])
                
                cv2.putText(image_seg,str(w)+"x"+str(h),(cX-15,cY),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                cv2.putText(image_seg, colors.getName(color_index) , (cX-15, cY+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        
    cv2.imshow("segmented",image_seg)
cam.stop()
cv2.destroyAllWindows()
lights.stop() 