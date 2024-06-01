#-------------------------------------#
#   Use camera or video for detection
#   Just use the camera and run it directly
#   When using video, you can specify the path to cv2.VideoCapture()
#-------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from ssd import SSD

ssd = SSD()
#-------------------------------------#
#   Using camera
#   capture=cv2.VideoCapture("1.mp4")
#-------------------------------------#
capture=cv2.VideoCapture(0)
fps = 0.0

while(True):
    t1 = time.time()
    # input a frame
    ref,frame=capture.read()
    # Format conversion, BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # Convert to Image
    frame = Image.fromarray(np.uint8(frame))
    # Perform the detection
    frame = np.array(ssd.detect_image(frame))
    # RGB to BGR to meet opencv display format
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)
    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break
