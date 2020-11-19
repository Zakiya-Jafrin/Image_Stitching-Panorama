import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

import CornerDetection as h
import FeatureDescriber as fea
import SIFT_auto as sift
import ransac as ran
import stitching as st
import harris as ha

import warnings
warnings.filterwarnings('ignore')

path = os.path.abspath('../project/project_images')
path_result = os.path.abspath('../project/results')

img_box = path + '/Boxes.png'
img_1 = path + '/Rainier1.png'
img_2 = path + '/Rainier2.png'
img_3 = path + '/Rainier3.png'
img_4 = path + '/Rainier4.png'
img_5 = path + '/Rainier5.png'
img_6 = path + '/Rainier6.png'

img_own_1 = path + '/1.jpg'
img_own_2 = path + '/2.jpg'
img_own_3 = path + '/3.jpg'

img_extra_11 = path + '/Hanging1.png'
img_extra_12 = path + '/Hanging2.png'

img_extra_31 = path + '/ND1.png'
img_extra_32 = path + '/ND2.png'

imgBox = cv2.imread(img_box)
img1 = cv2.imread(img_1)
img2 = cv2.imread(img_2)
img3 = cv2.imread(img_3)
img4 = cv2.imread(img_4)
img5 = cv2.imread(img_5)
img6 = cv2.imread(img_6)

imharris1 = cv2.imread(img_1)
imharris2 = cv2.imread(img_2)

own1 = cv2.imread(img_own_1)
own2 = cv2.imread(img_own_2)
own3 = cv2.imread(img_own_3)

imgex11 = cv2.imread(img_extra_11)
imgex12 = cv2.imread(img_extra_12)

imharris11 = cv2.imread(img_extra_11)
imharris12 = cv2.imread(img_extra_12)

imgex31 = cv2.imread(img_extra_31)
imgex32 = cv2.imread(img_extra_32)

imharris31 = cv2.imread(img_extra_31)
imharris32 = cv2.imread(img_extra_32)

# Question 1 : detect corners
kp, mg, th, keypoint_box = h.Corner(imgBox, 2)
keypoints1, image1_magnitude, image1_orientations, keypoint_image1 = h.Corner(imharris1,.5)
keypoints2, image2_magnitude, image2_orientations, keypoint_image2 = h.Corner(imharris2,.5)

cv2.imshow("keypoint_box", keypoint_box)
cv2.imshow("keypoint_image1", keypoint_image1)
cv2.imshow("keypoint_image2", keypoint_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(path_result+'/1a'+'.png',keypoint_box.astype(np.uint8))
cv2.imwrite(path_result+'/1b'+'.png',keypoint_image1.astype(np.uint8))
cv2.imwrite(path_result+'/1c'+'.png',keypoint_image2.astype(np.uint8))

# EXTRA CREDIT : roating Keypoint Detect
import SiftKeypointDetector as newDetect

o, image11_magnitude, image11_orientations, keypoint_image11 = h.Corner(imharris11, .5) #.5, 1.1
p, image12_magnitude, image12_orientations, keypoint_image12 = h.Corner(imharris12,.5)
keypoints11 = newDetect.new_impl(imgex11)
keypoints12 = newDetect.new_impl(imgex12)

# EXTRA CREDIT : Old_port Keypoint Detect
import SiftKeypointDetector as newDetect

o, image31_magnitude, image31_orientations, keypoint_image31 = h.Corner(imharris31, .5) #.5, 1.1
p, image32_magnitude, image32_orientations, keypoint_image32 = h.Corner(imharris32,1.1)
keypoints31 = newDetect.new_impl(imgex31)
keypoints32 = newDetect.new_impl(imgex32)

# Question 2 : describe features and draw keypoints
key1, key2, good, matched_keypoints = fea.runDescriptor(imharris1, keypoints1, image1_magnitude, image1_orientations,imharris2, keypoints2, image2_magnitude, image2_orientations, .7)
cv2.imwrite(path_result+'/2'+'.png',matched_keypoints.astype(np.uint8))

# # Question 3: run the Ransac algorithm to compute homography, draw the inliers
good_matches,kp1,kp2 = sift.match(img1,img2)
hom, homInv,inliersFinal  = ran.RANSAC(good_matches,800,.9,kp1,kp2)
im_inlier = cv2.drawMatches(img1,kp1, img2, kp2, inliersFinal, None, flags=2)
cv2.imwrite(path_result+'/3.png',im_inlier.astype(np.uint8))
cv2.imshow("Inliers", im_inlier)
cv2.waitKey(0)
cv2.destroyAllWindows()


# #  Question 4: Stitch the two images based on the inliers given by the ransac algorithm
i =0
pano = []
panorama, blend1,blend2= st.stitch(img1, img2, hom, homInv)
pano.append(panorama.astype(np.uint8))
cv2.imwrite(path_result+'/4.png',panorama.astype(np.uint8))
cv2.imshow("Stitched", pano[0])
cv2.waitKey(0)
cv2.destroyAllWindows()


# Question: E1 : Stitch all the images of rainer and create a big panorama
imgs = [img1, img2, img3, img4, img5, img6]
for m in range(len(imgs)):
    if(m<(len(imgs)-2)):
        good_matches,kp1,kp2 = sift.match(pano[m],imgs[m+2])
        hom, homInv,inliersFinal  = ran.RANSAC(good_matches,400,.7,kp1,kp2)
        panorama, b1,b2 = st.stitch(pano[m],imgs[m+2], hom, homInv)
        pano.append(panorama.astype(np.uint8))
        cv2.imwrite(path_result+'/steps_stitched%d.jpg' % (i),panorama.astype(np.uint8))
        i+=1
cv2.imwrite(path_result+'/Allstitched.png' ,pano[4].astype(np.uint8))
cv2.imshow("AllStitched", pano[4].astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Question :2 Self taken picture stitcihng of my yard , this onetakes 10-12 mins to run sometimes. 
# So this function is called at the very END.
def own_stitching_run():
    
    kpo1 ,ds1= sift.harris_own(own1)
    kpo2 ,ds2= sift.harris_own(own2)
    kpo3 ,ds3= sift.harris_own(own3)
    
    imgo1 = cv2.imread(img_own_1)
    imgo2 = cv2.imread(img_own_2)
    imgo3 = cv2.imread(img_own_3)
 
    good_matches_o = sift.match_own(imgo1,imgo2, kpo1, kpo2)
    hom_o, homInv_o,inliersFinal_o  = ran.RANSAC(good_matches_o,100,.2,kpo1,kpo2)
    pano_o, b1,b2= st.stitch(own1, own2, hom_o, homInv_o)
    cv2.imwrite(path_result+'/ownstep2.jpg',pano_o.astype(np.uint8))
    
    a = cv2.imread(path_result + '/ownstep2.jpg')
    kpos ,ds4 = sift.harris_own(a)
    own = cv2.imread(path_result + '/ownstep2.jpg')
    good_matches_o2 = sift.match_own(own ,imgo3, kpos, kpo3)
    hom_o2, homInv_o2,inliersFinal_o2  = ran.RANSAC(good_matches_o2,300,.2,kpos,kpo3)
    pano_o2, b1,b2= st.stitch(own, own3, hom_o2, homInv_o2)
    cv2.imwrite(path_result+'/ownStitched.png',pano_o2.astype(np.uint8))
    
    cv2.imshow("ownStitched", pano_o2.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#  Question 4 :EXtra blending two images Using laplacian pyramid
smooth_blending = st.blending (blend1,blend2)
cv2.imwrite(path_result+'/BlendedImage.png',smooth_blending.astype(np.uint8))
cv2.imshow("Blended", smooth_blending.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

#  Question 5 :EXTRA old Port IMAGE Stitching using SIFT detector(implemented)
key31, key32, good3, matched_keypoints3 = fea.runDescriptor(imgex31, keypoints31, image31_magnitude, image31_orientations,
              imgex32, keypoints32, image32_magnitude, image32_orientations, .75)
cv2.imwrite(path_result+'/old_matching'+'.png',matched_keypoints3.astype(np.uint8))

hom3, homInv3,inliersFinal3  = ran.RANSAC(good3,600,.7,key31,key32)
image13 = cv2.drawMatches(imgex31,key31, imgex32, key32, inliersFinal3, None, flags=2)
cv2.imwrite(path_result+'/old_matching_point_inliers'+'.png',image13.astype(np.uint8))
cv2.imshow("old_matching_point_inliers", image13)
cv2.waitKey(0)
cv2.destroyAllWindows()

## with the same values of thrshold and iterations the Stitched image sometimes becomes a bit different form each other,
##issue can be solved by running the previous block (where ransac is calculating the homography) if the number of iteration is increase the probablity of giving wrong result increases. so with the same value run sevral times the correct stiched image can be seen. also the sample produced outputs are given in the result directory (that expected)
## i suggest running it from Final.ipynb(second last block)
panorama3, _,_= st.stitch(imgex31, imgex32, hom3, homInv3)
cv2.imwrite(path_result+'/old_stitched.png',panorama3.astype(np.uint8))
cv2.imshow("old_stitched", panorama3.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


## Question 3 :EXTRA Rotated IMAGE Stitching using SIFT detector(implemented)
key11, key12, good1, matched_keypoints1 = fea.runDescriptor(imgex11, keypoints11, image11_magnitude, image11_orientations,
              imgex12, keypoints12, image12_magnitude, image12_orientations, .9)
good1, key11, key12 = sift.match(imgex11,imgex12)
cv2.imwrite(path_result+'/rotated_matching'+'.png',matched_keypoints1.astype(np.uint8))

hom1, homInv1,inliersFinal1  = ran.RANSAC(good1,1000,.09,key11,key12)
image11 = cv2.drawMatches(imgex11,key11, imgex12, key12, inliersFinal1, None, flags=2)
cv2.imwrite(path_result+'/rotated_matching_point_inliers'+'.png',image11.astype(np.uint8))
cv2.imshow("rotated_matching_point_inliers", image11)
cv2.waitKey(0)
cv2.destroyAllWindows()

panorama1, _,_= st.stitch(imgex11, imgex12, hom1, homInv1)
cv2.imwrite(path_result+'/rotated_stitched.png',panorama1.astype(np.uint8))
cv2.imshow("rotated_stitched", panorama1.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# NOTE: THIS ONE TAKES ABOUT 6-10 MINS TO PROVIDE THE OUTPUT
own_stitching_run() 

# img_result1a,_= ha.corner(imgBox)
# img_result1b,_= ha.corner(imharris1)
# img_result1c,_= ha.corner(imharris2)
# cv2.imwrite(path_result+'/1a'+'.png',img_result1a.astype(np.uint8))
# cv2.imwrite(path_result+'/1b'+'.png',img_result1b.astype(np.uint8))
# cv2.imwrite(path_result+'/1c'+'.png',img_result1c.astype(np.uint8))


