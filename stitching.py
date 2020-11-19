import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

import ransac as rs

def stitch(img1, img2, hom, homInv):
    h1, w1 = img1.shape[:2]
    height, width = img2.shape[:2]
    corners =[]
    
    corners.append(rs.project(0,0, homInv))
    corners.append(rs.project(0, height - 1, homInv))
    corners.append(rs.project(width - 1, height - 1, homInv))
    corners.append(rs.project(width - 1, 0, homInv))
    
#     result = cv2.warpPerspective(img2, homInv,(img2.shape[1] + img1.shape[1], img2.shape[0]))
#     cv2.imshow("Scanned Image", result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    corners = np.float32( np.array(corners))
    
    (minX, maxX, minY, maxY )= (0, w1, 0, h1)
    
    for l in range(len(corners)):
        if ((corners[l][0]) < minX):
            minX = math.floor(corners[l][0])
        elif((corners[l][0]) > maxX):
            maxX = math.ceil(corners[l][0])
            
        if((corners[l][1]) < minY):
            minY = math.floor(corners[l][1])
        elif((corners[l][1]) > maxY):
            maxY = math.ceil(corners[l][1])

    minX = abs(minX)
    minY = abs(minY)
    
    stitchedRow = (minY + maxY)
    stitchedCol = (maxX + minX)

    stitched = np.zeros([stitchedRow,stitchedCol, 3])
    blendImg1 = np.zeros(stitched.shape)
    blendImg2 = np.zeros(stitched.shape)
    temp = np.zeros(stitched.shape)

    
    for h in range(h1):
        for w in range(w1):
            if img1[h][w][0] != 0 or img1[h][w][1] != 0 or img1[h][w][2] != 0:
                stitched[h+minY][w + minX]  = img1[h][w]
                temp[h+minY][w + minX]  = img1[h][w]
                
    for a in range(stitchedRow):
        for b in range(stitchedCol):
            xs , ys = rs.project(b-minX, a-minY, hom)
            xs = xs.astype(int)
            ys = ys.astype(int)
            if((temp[a][b]).all() !=0):
                blendImg1[a][b] = temp[a][b]
            
            if(xs > 0 and xs < width and ys > 0 and ys < height):
                stitched[a][b] = img2[ys][xs] 
                stitched[a][b] = cv2.getRectSubPix(img2, (1,1), (xs,ys)) 
                if((stitched[a][b]).all() !=0):
                    blendImg2[a][b] = stitched[a][b] 
    
    return stitched, blendImg1, blendImg2

def blending(img1, img2):
    levels = 6
    gaussianImg1 = [img1.astype(np.float32)]
    gaussianImg2 = [img2.astype(np.float32)]
    for i in range(levels):
        img1 = cv2.pyrDown(img1).astype(np.float32)
        gaussianImg1.append(img1)
        img2 = cv2.pyrDown(img2).astype(np.float32)
        gaussianImg2.append(img2)
    laplacianImg1 = [gaussianImg1[levels]]
    laplacianImg2 = [gaussianImg2[levels]]

    for i in range(levels,0,-1):
        temp = cv2.pyrUp(gaussianImg1[i]).astype(np.float32)
        temp = cv2.resize(temp, (gaussianImg1[i-1].shape[1],gaussianImg1[i-1].shape[0]))
        laplacianImg1.append(gaussianImg1[i-1]-temp)

        temp = cv2.pyrUp(gaussianImg2[i]).astype(np.float32)
        temp = cv2.resize(temp, (gaussianImg2[i-1].shape[1],gaussianImg2[i-1].shape[0]))
        laplacianImg2.append(gaussianImg2[i-1]-temp)

    laplacianList = []
    for lpImg1,lpImg2 in zip(laplacianImg1,laplacianImg2):
        rows,cols = lpImg1.shape[:2]
        mask1 = np.zeros(lpImg1.shape,dtype = np.float32)
        mask2 = np.zeros(lpImg2.shape,dtype = np.float32)
        mask1[:, 0:int(cols/ 2)] = 1
        mask2[:,int( cols / 2):] = 1

        temp1 = lpImg1 * mask1
        temp2 = lpImg2 * mask2
        temp = temp1 + temp2
        
        laplacianList.append(temp)
    
    blend = laplacianList[0]
    for i in range(1,levels+1):
        blend = cv2.pyrUp(blend)   
        blend = cv2.resize(blend, (laplacianList[i].shape[1],laplacianList[i].shape[0]))
        blend = blend+ laplacianList[i]
    
    np.clip(blend, 0, 255, out=blend)
    crop = blend[64:(64+len(blend)), 0:(0+len(blend[0]))]
    return crop

# blending (blend1,blend2)
