import cv2
from cv2 import Sobel
import numpy as np
import os
import glob

filepath=r"C:\Users\user\Desktop\hw2.5.1.png"

# images = glob.glob(filepath+"\*.bmp")


imgL = cv2.imread(r'C:\Users\user\Desktop\OpenCVDL\hw2\Q4_Image\imL.png',0)
imgR = cv2.imread(r'C:\Users\user\Desktop\OpenCVDL\hw2\Q4_Image\imR.png',0)


stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
disparity = stereo.compute(imgL, imgR)
disp=disparity
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
disparity = cv2.resize(disparity, (0, 0), fx=0.25, fy=0.25)

imgL_disparity = cv2.resize(imgL, (0, 0), fx=0.25, fy=0.25)
imgR_disparity = cv2.resize(imgR, (0, 0), fx=0.25, fy=0.25)
print(disp)
cv2.imshow('image', disparity)



imgL = cv2.imread(r'C:\Users\user\Desktop\OpenCVDL\hw2\Q4_Image\imL.png')
imgR = cv2.imread(r'C:\Users\user\Desktop\OpenCVDL\hw2\Q4_Image\imR.png')
imgL_disparity = cv2.resize(imgL, (0, 0), fx=0.25, fy=0.25)
imgR_disparity = cv2.resize(imgR, (0, 0), fx=0.25, fy=0.25)
point=(100,100)


def draw(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        imgL=imgL_disparity.copy()
        imgR_C=imgR.copy()
        cv2.circle(imgL, (x,y), 1, (0,0,255), 7)
        if disparity[y][x]!=0:
            cv2.circle(imgR_C, (4*x-(disparity[y][x]),y*4), 1, (0,0,255), 28)
        imgR_r = cv2.resize(imgR_C, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow('imgL', imgL)
        cv2.imshow('imgR', imgR_r)
        # print(disp[y][x])


cv2.imshow('imgL', imgL_disparity)
cv2.imshow('imgR', imgR_disparity)
cv2.setMouseCallback('imgL', draw)


cv2.waitKey(0)
cv2.destroyAllWindows()