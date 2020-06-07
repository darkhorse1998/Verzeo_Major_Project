# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:51:06 2020

@author: IMPOSSIBLE
"""

from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
from imutils.video import VideoStream

vs = VideoStream(src=0).start()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
while(True):
        
    image = vs.read()
    image = imutils.resize(image, width=min(400, image.shape[1]))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.20)

    
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("output", image)
    if(cv2.waitKey(1) & 0xFF == ord('q')):break
vs.stream.release()
cv2.destroyAllWindows()
