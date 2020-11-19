run:
--go to the directory
--python main.py would run the whole code.
--to support the evaluation i have provided an well described Final.ipynb file
--the results should be saved in the folder called result in the project directory
-- in the results directory a folder named 'Produced_result" is given to explain what to expect after running the program


FEATURE DETECTION AND MATCHING: 
------1. Compute the Harris corner detector using the following steps
            a.Compute the x and y derivatives on the image
            b. Compute the covariance matrix H of the image derivatives. 
            c. Compute the Harris response using determinant(H)/trace(H).
         
           ## The implementation of this Harris corner is done in CornerDetection.py, the Corner() function is called to detect the keypoints which takes an image as param and return the keypoints, and the magnitude and orientation(to be used in descriptor)
           and after detecting the corners the files of the Boxes.png, Rainier1.png and Rainier2.png is saved in the result folder as 1a.png, 1b.png and 1c.png

-----2. Matching the interest points between two images.
            a. Compute the descriptors for each interest point.
            b. For each interest point in image 1, find its best match in image 2.
            c. Add the pair of matching points to the list of matches
            
            ###The Implementation of feature descriptor is done in FeatureDescriber.py the final function runDescriptor()
takes two images and their corresponding keypoints, the magnitude and the orientation of the keypoints and finally retirns the matches. the result for Rainier1.png and Rainier2.png is saved as 2.png


From this point opncv sift library is used, except for the extra credit parts
-----3: Compute the homography between the images using RANSAC:
            a. function project(x1, y1, H). This should project point (x1, y1) using thehomography “H”. Return the projected point (x2, y2). 
            
            b. Function computeInlierCount(H, matches, inlierThreshold, keypoint1, keypoint2). that computes the number of inlying points given a homography "H". That is, project the first point in each match using the function "project". If the projected point is less than the distance "inlierThreshold" from the second point, it is an inlier. Return the total number of inliers.
            
            C. Function RANSAC (matches, numIterations, inlierThreshold,keypoint1, keypoint2). This function takes a list of potentially atching points between two images and returns the homography transformation that  return hom, homInv and the finalinliers
            
NOTE: the number of inliers seems to be a bit higher because, the matching is calculated using self implemntation. After that I again implented the opencv library to compute the inliers. I did this because own impelmntation was taking more time to execute.

            ##The implementation is done is done in ransac.py and the inliers are saved as 3.png (for this implementation opencv is implemented - as own implementation takes a lot of time. the functions are implemented in SIFT_auto.py)
            
-------4. the function stitch(image1, image2, hom, homInv). 
            a. Compute the size of "stitchedImage."project the four corners of "image2" onto "image1" and "homInv"
            b. Copy "image1" onto the "stitchedImage" at the right location.
            c. For each pixel in "stitchedImage", project the point onto "image2". If it lies within image2's boundaries, add or blend the pixel's value to "stitchedImage. " 
            
          ## the final stitched the images of "Rainier1.png" and "Rainier2.png". Save the stitched image as "4.png". 
          
GRADUATE COMPULSORY:      
         
---------1. a panorama that stitches together the six Mt. Rainier photographs, i.e. , Rainier1.png, ... Painier6.png. The final result should look similar to "AllStitched.png". this is done in main.py

---------2. own panorama using three or more images. this is done in the own_stitching_run() in the main. However, for this one I implemented harris and the features are detected using sift only.
NOTE: this one takes approximately larger time to exectue

EXTRA CREDIT:

-------3. new image descriptor that can stitch the images "Hanging1.png" and "Hanging2.png". Save the Stitched image and the "match" images. For this implementation in NEW detector. SIFT detector in SiftKeypointDetector.py the final function new_impl return the keypoints. This detectes the enough mathching and stitches the two images and these are saved as hanging_matching.png, hanging_matching_point_inliers.png and Hanging _stitched.png. (the best result is found with 1000 iterations and inlierthreshold .09 and the ratio is .9)

-------4. In the stitch.py a blending() function implemnted which takes two positonal images as parameters and stitch them using laplacian pyramid. the image is saved as BlendedImage.png

-------5. Implement a new image detector, SIFT detector in SiftKeypointDetector.py the final function new_impl return the keypoints. This detectes the enough mathching and stitches the two images with the previous that is in featureDescriptor.py implementation of descriptor sttiches these images. that can stitch the images ND1.png and ND2.png. the matches, inliers and stitched image is saved as old_matching.png, old_matching_point_inliers.png and old_stitched.png (the best result is found with 600 iterations and inlierthreshold .6 and the ratio is .75)
NOTE: with the same values of thrshold and iterations the Stitched image sometimes becomes a bit different form each other, issue can be solved by running the previous block (where ransac is calculating the homography) if the number of iteration is increase the probablity of giving wrong result increases. so with the same value run sevral times the correct stiched image can be seen. also the sample produced outputs are given in the result directory (that expected)


