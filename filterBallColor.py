import cv2
import numpy as np
from cvzone.ColorModule import ColorFinder  # for detect the ball color

colorFinder = ColorFinder(True) # by default is False, True kan supaya muncul trackbar nya

while True:
    img = cv2.imread('img.png')
    imgColor, mask = colorFinder.update(img)

    cv2.imshow('Image', img)
    cv2.imshow('Image Color', imgColor)
    cv2.waitKey(1)     # if you need to slow down, change from 1 to 50

'''
{'hmin': 32, 'smin': 50, 'vmin': 0, 'hmax': 45, 'smax': 255, 'vmax': 255}
'''