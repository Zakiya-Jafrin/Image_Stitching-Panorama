import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def match(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches=[]
    good_points=[]
    for m1, m2 in raw_matches:
        if m1.distance < .3 * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append(m1)
    return good_matches, kp1, kp2


def harris_own(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray_img.astype(np.float32), 2, 3, 0.04)
    result_img = img.copy() # deep copy image

    # Threshold for an optimal value, it may vary depending on the image.
    result_img[dst > 0.01 * dst.max()] = [0, 255, 0]

    # for each dst larger than threshold, make a keypoint out of it
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]

    return (keypoints, result_img)

def match_own(img1,img2, kp1, kp2):
    sift = cv2.xfeatures2d.SIFT_create()
    a, des1 = sift.compute(img1,kp1)
    b, des2 = sift.compute(img2,kp2)
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_points = []
    good_matches=[]
    for m1, m2 in raw_matches:
        if m1.distance < .34 * m2.distance:
            good_points.append(cv2.DMatch(m1.trainIdx, m1.queryIdx, 1))
            good_matches.append(m1)
    return good_matches