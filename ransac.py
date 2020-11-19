import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def project(x1, y1, H):
    w = (H[2][0]*x1 + H[2][1]*y1 + H[2][2])
    if (w == 0):
        w = 0.00000001
    x2 = np.float32(((H[0][0]*x1 + H[0][1]*y1 + H[0][2])/w))
    y2 = np.float32(((H[1][0]*x1 + H[1][1]*y1 + H[1][2])/w))
    return x2, y2

def computeInlierCount(H, matches, inlierThreshold, keypoint1, keypoint2): 
    numMatches = 0
    inliers=[]
    for m in range(len(matches)):
        x2, y2 = project(keypoint1[matches[m].queryIdx].pt[0], keypoint1[matches[m].queryIdx].pt[1], H)
        x1 = keypoint2[matches[m].trainIdx].pt[0]
        y1 = keypoint2[matches[m].trainIdx].pt[1]
        distance = math.sqrt(((x1-x2)**2) + ((y1-y2)**2))
        if(distance < inlierThreshold):
            numMatches+= 1
            inliers.append(matches[m])
    return numMatches, inliers

def RANSAC (matches, numIterations, inlierThreshold,  keypoint1, keypoint2):
    finalSrc =[]
    finalDst =[]
    maxNumInliers = 0
    for i in range(numIterations):
        srcPoints =[]
        dstPoints =[]
        four_random_matches = random.sample(matches, 4)
        
        for j in range(len(four_random_matches)):
            srcPoints.append(keypoint1[four_random_matches[j].queryIdx].pt)
            dstPoints.append(keypoint2[four_random_matches[j].trainIdx].pt)

        src = np.float32(srcPoints).reshape(-1,1,2)
        dst = np.float32(dstPoints).reshape(-1,1,2)
        
        H, status = cv2.findHomography(src, dst, 0)
        if(H is None):
            continue
        
        numMatches, inliers = computeInlierCount(H, matches, inlierThreshold, keypoint1, keypoint2)
        if (numMatches > maxNumInliers):
            maxNumInliers = numMatches
            hom = H
        
    numMatchesFinal, inliersFinal = computeInlierCount(hom, matches, inlierThreshold,keypoint1, keypoint2)
    
    for k in range(len(inliersFinal)):
        finalSrc.append(keypoint1[inliersFinal[k].queryIdx].pt)
        finalDst.append(keypoint2[inliersFinal[k].trainIdx].pt)
        
    srcF = np.float32(finalSrc).reshape(-1,1,2)
    dstF = np.float32(finalDst).reshape(-1,1,2)
        
    hom, status = cv2.findHomography(srcF, dstF, 0 )
    homInv = np.float32(np.linalg.inv(hom))
    
    return hom, homInv,inliersFinal