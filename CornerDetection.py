import cv2
import numpy as np
import math
import random

def Corner(img, THRESHOLD):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    image *= 1./255
    
    Ix = cv2.Sobel(image, -1,1,0,ksize=3)
    Iy = cv2.Sobel(image, -1,0,1,ksize=3)
    
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix *Iy
    
    Ix2 = cv2.GaussianBlur(Ix2, (3,3), 0)
    Iy2 = cv2.GaussianBlur(Iy2, (3,3), 0)
    Ixy = cv2.GaussianBlur(Ixy, (3,3), 0)
    
    determinant = Ix2*Iy2 - Ixy*Ixy
    trace= Ix2+Iy2
    response = np.divide(determinant,trace)
    response[np.isnan(response)]=0
    magnitude = cv2.sqrt(Ix**2+Iy**2)
    orientations=np.arctan2(Iy, Ix)
    
#     response = nonMaxSup(respons)
    keypoints = np.argwhere(response> THRESHOLD)
    keypoints = [cv2.KeyPoint(x[1],x[0], response[x[0],x[1]])for x in keypoints]
    outImage = cv2.drawKeypoints(img, keypoints,img)
    
#     keyp = adaptiveNonMaxSup(image, response, 4)
    
    return keypoints, magnitude, np.rad2deg(orientations), outImage


def adaptiveNonMaxSup(image, keypoints, num):
    keys_with_radius =[]
    final_result =[]
    for i in range(len(keypoints)):
        key1 = keypoints[i].pt
        lowest_radius = 9999999999
        for j in range(len(keypoints)):
            if(i==j):
                continue
            key2 = keypoints[j].pt
            radius = math.sqrt((key1[0]-key2[0])**2 + (key1[1]-key2[1])**2)
            if(radius < lowest_radius):
                lowest_radius = radius
        keys_with_radius.append([key1[0], key1[1], lowest_radius])
        
    keys_with_radius = sorted(keys_with_radius, key = lambda x :x[2], reverse = True)[:num]
    for key in keys_with_radius:
        final_result.append(cv2.KeyPoint(key[0], key[1], key[2]))
#     outImage = cv2.drawKeypoints(image, final_result,image)
    return final_result

def nonMaxSup(harris):
    harris = np.pad(harris, 1, mode='constant')
    rows, cols = harris.shape
    
    for i in range(1, rows-1, 1):
        for j in range(1, cols-1, 1):
            patch = harris[i-1: i+3, j-1:j+3]
            if(harris[i, j] != patch.max()):
                harris[i,j ]=0
    harris = harris[1: rows, 1:cols]
    return harris