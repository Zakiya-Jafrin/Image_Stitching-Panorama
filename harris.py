import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def HarrisCorner(grayImage, rows, cols):
    corners = np.zeros([rows, cols], dtype = np.float32)
    Ixx = np.zeros([rows, cols], dtype = np.float32)
    Ixy = np.zeros([rows, cols], dtype = np.float32)
    Iyy = np.zeros([rows, cols], dtype = np.float32)
    
  
    sobel_X = (np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32))
    sobel_Y = (np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32))

    dx = cv2.filter2D(grayImage,-1,sobel_X)
    dy = cv2.filter2D(grayImage,-1,sobel_Y)
    
    for i in range(len(grayImage)):
        for j in range(len(grayImage[0])):
            x = dx[i,j]
            y = dy[i,j]
            Ixx [i, j] = x*x 
            Ixy [i, j] = x*y
            Iyy [i, j] = y*y
    
#     print(len(kps))
    s_xx = cv2.GaussianBlur(Ixx, (3,3), 0)
    s_xy = cv2.GaussianBlur(Ixy, (3,3), 0)
    s_yy = cv2.GaussianBlur(Iyy, (3,3), 0)
    
    for i in range(rows):
        for j in range(cols):
            a = s_xx[i,j]
            b = s_xy[i,j]
            c = s_xy[i,j]
            d = s_yy[i,j]
            
            if (a + d == 0):
                cornerStrength = 0
            
            else: 
                det = (a*d)-(b*c)
                tr = a + d
                cornerStrength = det/tr
                
            if (cornerStrength > .08):
                corners[i,j] = cornerStrength
                
    return corners

def NonMaxSupp(img,grayImage,corners):
    height, width = grayImage.shape
    suppressed = np.zeros([height, width], dtype = np.float32)
    keyPoints =[]
    points =[]
    window_size = 3
    offset = int(window_size/2)
    maxRow =0
    maxCol =0
    
    for i in range(offset, height-offset):
        for j in range(offset, width-offset):
            max = 0
            for k in range (i-offset, i+offset) :
                for l in range (j-offset, j+offset) :
                    current = corners[i,j]
                    if (current > max) :
                        max = current
                        maxRow = k
                        maxCol = l
            if (max > 0) :
                suppressed[maxRow, maxCol] = max
                keyPoints.append((maxRow,maxCol))
                points.append(cv2.KeyPoint(maxRow,maxCol,1))
                
    img_copy_for_corners = np.copy(img)
    for rowindex, response in enumerate(suppressed):
        for colindex, a in enumerate(response):
            if a > 0:
                img_copy_for_corners[rowindex, colindex] = [0,0,255]
                
    cv2.imshow("corner", img_copy_for_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_copy_for_corners, points
    

def corner(img):
    rows = len(img)
    cols = len(img[0])
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grayImage *= 1./255
    harrisCorner = HarrisCorner(grayImage, rows, cols)
    img_result,p= NonMaxSupp(img,grayImage,harrisCorner) 
    return img_result, p