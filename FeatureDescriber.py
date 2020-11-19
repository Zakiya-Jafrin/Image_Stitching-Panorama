import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def getDominantAngle(patch, patch_magnitude):
    angles = np.zeros(36)
    for row in range(len(patch)):
        for col in range(len(patch[0])):
            cell_angle = patch[row, col]
            cell_magnitude = patch_magnitude[row, col]       
            index = int(cell_angle/10)
            angles[index-1] += cell_magnitude
    return angles.argmax()


def getDescriptor(interest_point, image_magnitude, image_orientations):
    x, y= interest_point.pt
    top = int(y-8)
    left = int(x-8)
    
    patch = image_orientations[top:top+16, left:left+16]
#     print(patch.shape)
    patch_magnitude = image_magnitude[top:top+16, left:left+16]
    dominant_angle = getDominantAngle(patch, patch_magnitude)
    patch = patch - dominant_angle
    
    feature_vectors = np.zeros(128)
    index =0
    
    for i in range (0,16,4):
        for j in range(0,16,4):
            subpatch = patch[i:i+4, j:j+4]
            bin_1 = ((subpatch>= -180)& (subpatch<-135)).sum()
            bin_2 = ((subpatch>= -135)& (subpatch<-90)).sum()
            bin_3 = ((subpatch>= -90)& (subpatch<-45)).sum()
            bin_4 = ((subpatch>= -45)& (subpatch<0)).sum()
            bin_5 = ((subpatch>= 0)& (subpatch<45)).sum()
            bin_6 = ((subpatch>= 45)& (subpatch<90)).sum()
            bin_7 = ((subpatch>= 90)& (subpatch< 135)).sum()
            bin_8 = ((subpatch>= 135)& (subpatch<180)).sum()
            feature_vectors[index:index+8] = bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,bin_7,bin_8
            
            index +=8
    return feature_vectors

def getDescriptorForImage(keypoints, magnitudes, oreintations):   
    image_descriptor=[]
    for i in range(len(keypoints)):
        descriptor = getDescriptor(keypoints[i], magnitudes, oreintations)
        image_descriptor.append(descriptor)
    return image_descriptor

def getDistance(feature_vector1, feature_vector2):
    return np.sum((feature_vector1-feature_vector2)**2)

def getSSD(image_descriptor1, image_descriptor2):
    ssd_values =[]
    for i in range(len(image_descriptor1)):
        desc1 = image_descriptor1[i]
        lowest = 999999999999
        lowest_index = -1
        for j in range(len(image_descriptor2)):
            desc2 = image_descriptor2[j]
            distance = getDistance(desc1, desc2)
            if(distance<lowest):
                lowest = distance
                lowest_index =j
        ssd_values.append([lowest, lowest_index])
    return ssd_values

def getSSD2(image_descriptor1, image_descriptor2, ssd_values):
    ssd_values2 =[]
    for i in range(len(image_descriptor1)):
        desc1 = image_descriptor1[i]
        lowest = 999999999999
        lowest_index = -1
        for j in range(len(image_descriptor2)):
            desc2 = image_descriptor2[j]
            distance = getDistance(desc1, desc2)
            if(distance<lowest and distance != ssd_values[i][0]):
                lowest = distance
                lowest_index =j
        ssd_values2.append([lowest, lowest_index])
    return ssd_values2

def contrastHandler(desc):
    normalized_list =[]
    for d in desc:
        d = d/(d.sum()+.000000001)
        d = d.clip(0,0.2)
        d = d/d.sum()
        normalized_list.append(d)
    return normalized_list

def ratioTest(lowest_ssd, lowest_ssd2, ratio):
    ratio_values=[]
    for i in range(len(lowest_ssd)):
        ssd1 = lowest_ssd[i][0]
        ssd2 = lowest_ssd2[i][0]       
        if(ssd1/ssd2 <= ratio):
            ratio_values.append([ratio, i, lowest_ssd[i][1]])
    return ratio_values

def rotationInvariance(patch, patch_magnitude):
    angles = np.zeros(18)
    for row in range(len(patch)):
        for col in range(len(patch[0])):
            cell_angle = patch[row, col]
            cell_magnitude = patch_magnitude[row, col]       
            index = int(cell_angle/20)
            angles[index-1] += cell_magnitude
    return angles.argmax()


def runDescriptor(img1, kp1, mag1, orien1,img2, kp2, mag2, orien2, RATIO):
    desc1 = getDescriptorForImage(kp1, mag1, orien1)
    desc2 = getDescriptorForImage(kp2, mag2, orien2)
    image1_desc = contrastHandler(desc1)
    image2_desc = contrastHandler(desc2)
    ssd_values = getSSD(image1_desc, image2_desc)
    ssd_second_values = getSSD2(image1_desc, image2_desc, ssd_values)
    ratio_values = ratioTest(ssd_values, ssd_second_values, RATIO)
    matching_points_ratio =[]
    for i in range(len(ratio_values)):
        matching_points_ratio.append(cv2.DMatch(ratio_values[i][1], ratio_values[i][2], ratio_values[i][0]))

    image3 = cv2.drawMatches(img1, kp1, img2, kp2, matching_points_ratio, None, flags=2)
    cv2.imshow("Matching_points", image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return kp1, kp2 , matching_points_ratio, image3
