# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:32:15 2020

@author: IMPOSSIBLE
"""

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2
import os
import time

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

imagePaths = list(paths.list_images('images'))

for i,imagePath in enumerate(imagePaths):
    
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 255), 2)
    try:
        os.mkdir('output')
    except FileExistsError:
        if i == 0:
            print("Directory exists")

    filename = "output/person ("+str(i+1)+")_output.jpg"
    print("...Evaluating for image "+str(i+1)+"...")
    for i in range(29):
        print(".", end="")
        time.sleep(0.01)
    print("\n")
    cv2.imwrite(filename,image)
print("="*42)
print("Successfully Completed")
print("Please check the 'output' folder")