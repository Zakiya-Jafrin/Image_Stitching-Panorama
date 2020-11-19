import cv2
import numpy as np
import math
import random

import numpy.linalg as lin

import warnings
warnings.filterwarnings('ignore')

def filter_g(sigma): 
    size = 2*np.ceil(3*sigma)+1 
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1] 
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()


def octave_g(init_level, s, sigma): 
    octave = [init_level] 
    k = 2**(1/s) 
    kernel = filter_g(k * sigma)
    for _ in range(s+2): 
        next_level = cv2.filter2D(octave[-1], -1,kernel)
        octave.append(next_level) 
    return octave

def pyramid_g(im, num_octave, s, sigma): 
    pyr = [] 
    for _ in range(num_octave): 
        octave = octave_g(im, s, sigma) 
        pyr.append(octave) 
        im = octave[-3][::2, ::2] 
    return pyr

def generate_octave(gaussian_octave): 
    octave = [] 
    for i in range(1, len(gaussian_octave)):   
        octave.append(gaussian_octave[i] - gaussian_octave[i-1]) 
    return np.concatenate([o[:,:,np.newaxis] for o in octave], axis=2) 
    
def pyramid(pyramid_g): 
    pyr = [] 
    for gaussian_octave in pyramid_g: 
        pyr.append(generate_octave(gaussian_octave)) 
    return pyr

def get_candidate_keypoints(D, w=16): 
    candidates = [] 
    D[:,:,0] = 0 
    D[:,:,-1] = 0 
    for i in range(w//2+1, D.shape[0]-w//2-1): 
        for j in range(w//2+1, D.shape[1]-w//2-1): 
            for k in range(1, D.shape[2]-1): 
                patch = D[i-1:i+2, j-1:j+2, k-1:k+2] 
                if np.argmax(patch) == 13 or np.argmin(patch) == 13: 
                    candidates.append([i, j, k]) 
    return candidates

def localize_keypoint(D, x, y, s): 
    dx = (D[y,x+1,s]-D[y,x-1,s])/2. 
    dy = (D[y+1,x,s]-D[y-1,x,s])/2. 
    ds = (D[y,x,s+1]-D[y,x,s-1])/2. 
    dxx = D[y,x+1,s]-2*D[y,x,s]+D[y,x-1,s] 
    dxy = ((D[y+1,x+1,s]-D[y+1,x-1,s]) -(D[y-1,x+1,s]-D[y-1,x-1,s]))/4. 
    dxs = ((D[y,x+1,s+1]-D[y,x-1,s+1]) -(D[y,x+1,s-1]-D[y,x-1,s-1]))/4. 
    dyy = D[y+1,x,s]-2*D[y,x,s]+D[y-1,x,s] 
    dys = ((D[y+1,x,s+1]-D[y-1,x,s+1]) -(D[y+1,x,s-1]-D[y-1,x,s-1]))/4. 
    dss = D[y,x,s+1]-2*D[y,x,s]+D[y,x,s-1] 
    J = np.array([dx, dy, ds]) 
    HD = np.array([ [dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]]) 
    offset = -lin.inv(HD).dot(J)
    return offset, J, HD[:2,:2], x, y, s

def find_keypoints_for_DoG_octave(D, R_th, t_c, w): 
    candidates = get_candidate_keypoints(D, w)
    keypoints = [] 
    for i, cand in enumerate(candidates): 
        y, x, s = cand[0], cand[1], cand[2] 
        offset, J, H, x, y, s = localize_keypoint(D, x, y, s) 
        contrast = D[y,x,s] + .5*J.dot(offset) 
        if abs(contrast) < t_c: continue 
        w, v = lin.eig(H) 
        r = w[1]/w[0] 
        R = (r+1)**2 / r 
        if R > R_th: continue 
        kp = np.array([x, y, s]) + offset
        keypoints.append(kp)
    return np.array(keypoints)

def get_keypoints(DoG_pyr, R_th, t_c, w): 
    kps = [] 
    for D in DoG_pyr: 
        kps.append(find_keypoints_for_DoG_octave(D, R_th, t_c, w))
    return kps

def cart_to_polar_grad(dx, dy):
    m = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx)+np.pi) * 180/np.pi
    return m, theta

def get_grad(L, x, y):
    dy = L[min(L.shape[0]-1, y+1),x] - L[max(0, y-1),x]
    dx = L[y,min(L.shape[1]-1, x+1)] - L[y,max(0, x-1)]
    return cart_to_polar_grad(dx, dy)

def quantize_orientation(theta, num_bins):
    bin_width = 360//num_bins
    return int(np.floor(theta)//bin_width)

def fit_parabola(hist, binno, bin_width):
    centerval = binno*bin_width + bin_width/2.

    if binno == len(hist)-1: rightval = 360 + bin_width/2.
    else: rightval = (binno+1)*bin_width + bin_width/2.

    if binno == 0: leftval = -bin_width/2.
    else: leftval = (binno-1)*bin_width + bin_width/2.
    
    A = np.array([
        [centerval**2, centerval, 1],
        [rightval**2, rightval, 1],
        [leftval**2, leftval, 1]])
    b = np.array([
        hist[binno],
        hist[(binno+1)%len(hist)], 
        hist[(binno-1)%len(hist)]])

    x = lin.lstsq(A, b, rcond=None)[0]
    if x[0] == 0: x[0] = 1e-6
    return -x[1]/(2*x[0])


def new_impl(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    im = cv2.filter2D(image, -1,filter_g(1.3))
    gaussian_pyr = pyramid_g(im, 4, 3, 1.6)
    pyr = pyramid(gaussian_pyr)
    kp_pyr = get_keypoints(pyr, 10, 3, 16)
    point=[]
    for i, DoG_octave in enumerate(pyr):
        for j in range(len(kp_pyr[i])):
            point.append(cv2.KeyPoint(kp_pyr[i][j][0], kp_pyr[i][j][1],1))
            
            
    return point
