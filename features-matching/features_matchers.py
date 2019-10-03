"""
Manil Bastola
"""

import os
import sys
import cv2
import _pickle as pickle
#import cPickle as pickle
import math
import numpy as np
#from scipy.interpolate import griddata
from skimage.transform import PiecewiseAffineTransform, PolynomialTransform, warp

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,point.class_id, descriptors[i])
        i+=1
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []

    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

def computeDescriptions(img, featsfilename = None):
    if (featsfilename and os.path.exists(featsfilename)):
        feats = pickle.load( open(featsfilename , "rb" ) )
        return unpickle_keypoints(feats)
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(0,3,0.07)

    #compute keypoint & descriptions
    kp1, des1 = sift.detectAndCompute(img,None)
    if featsfilename:
        temp = pickle_keypoints(kp1, des1)
        pickle.dump(temp, open(featsfilename, "wb"))
        out=cv2.drawKeypoints(img, kp1,  0, color=(0,255,0), flags=0)
        cv2.imwrite( featsfilename+"_kps.png", out);
    return kp1, des1


def genMatcher():
    #use FLANN_INDEX_KDTREE algorithm
    flann_params = dict(algorithm = 1, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    return matcher

def similarityScore(src, dst, mask):
    dst_masked =  dst * mask
    #for viz
    diff = cv2.absdiff(dst,src)
    cv2.imwrite("difference.jpg",diff)            
    score = cv2.matchTemplate(src, dst,cv2.TM_CCORR_NORMED, None, mask)
    return score


def main():

    MIN_MATCH_COUNT = 10
    HOMOGRAPHY = 1
    OPTICAL_FLOW = 1
    
    if len(sys.argv) != 3:
        print("needed src and destination image paths")
        return

    src_image_path = sys.argv[1]
    dst_image_path = sys.argv[2]
    src_image = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    dst_image = cv2.imread(dst_image_path, cv2.IMREAD_COLOR)

    src_gray = cv2.cvtColor(src_image , cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst_image , cv2.COLOR_BGR2GRAY)
    
    kp1, desc1 = computeDescriptions(src_gray, "feats/{}_feats.kp".format(os.path.basename(src_image_path).split(".")[0]))
    kp2, desc2 = computeDescriptions(dst_gray, "feats/{}_feats.kp".format(os.path.basename(dst_image_path).split(".")[0]))
    
    matcher = genMatcher()
    knn_matches = matcher.knnMatch(desc1, desc2, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        if HOMOGRAPHY:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w,d = src_image.shape
            h2,w2,d2 = dst_image.shape
            
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            
            src_transformed = cv2.warpPerspective(src_image, M, (w2, h2))
            boundingRect = list(cv2.boundingRect(dst))
            if boundingRect[0]<0:
                boundingRect[0]=0
            if boundingRect[1]<0:
                boundingRect[1]=0
            if boundingRect[2] >= w2:
                boundingRect[2] = w2-1
            if boundingRect[3] >= h2:
                boundingRect[3] = h2-1            
            src_transformed_cropped = src_transformed[boundingRect[1]:boundingRect[1]+boundingRect[3], boundingRect[0]:boundingRect[0]+boundingRect[2]]
            dst_cropped = dst_image[boundingRect[1]:boundingRect[1]+boundingRect[3], boundingRect[0]:boundingRect[0]+boundingRect[2]].copy()

            src_transformed_cropped_gray = cv2.cvtColor(src_transformed_cropped , cv2.COLOR_BGR2GRAY)
            dst_cropped_gray = cv2.cvtColor(dst_cropped , cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(src_transformed_cropped_gray, 0, 1, cv2.THRESH_BINARY)        
            mask = cv2.merge((mask, mask, mask))
            score = similarityScore(src_transformed_cropped, dst_cropped, mask)

            print("Similarity Score: {}".format(score))

            cv2.imwrite("src_transfromed.jpg",src_transformed_cropped)

            dst_image = cv2.polylines(dst_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            
            #weighted blend for vizualization
            cv2.imwrite("homography_transform",0.5*src_transformed+0.5*dst_image)
            cv2.imwrite("dst_overlay.jpg",0.5*src_transformed_cropped+0.5*dst_cropped)

        if OPTICAL_FLOW:

            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (100,100),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS |cv2.TERM_CRITERIA_COUNT , 30, 0.1))
            
            # params for ShiTomasi corner detection
            feature_params = dict( maxCorners = 2000,
                                   qualityLevel = 0.05,
                                   minDistance = 3,
                                   blockSize = 3)

            color = np.random.randint(0,255,(100,3))
            p0 = cv2.goodFeaturesToTrack(dst_cropped_gray, mask = None, **feature_params)
            #_, p0 = computeDescriptions(src_transformed_cropped)
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(dst_cropped, src_transformed_cropped, p0, None, **lk_params)
            # Select good points
            good_old = p0[st==1]
            good_new = p1[st==1]


            src_oflow_viz = src_transformed_cropped.copy()
            dst_oflow_viz = dst_cropped.copy()
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                try:
                    a,b = new.ravel()
                    c,d = old.ravel()
                    src_oflow_viz = cv2.line(src_oflow_viz, (a,b),(c,d), color[i].tolist(), 2)
                    dst_oflow_viz = cv2.circle(dst_oflow_viz,(a,b),5,color[i].tolist(),-1)
                except:
                    continue
                
            cv2.imwrite("tmp3.jpg",src_oflow_viz);
            cv2.imwrite("tmp4.jpg",dst_oflow_viz);


            tform = PiecewiseAffineTransform()
            tform.estimate(good_old,good_new)
            
            out = (255*warp(src_transformed_cropped, tform)).astype(np.uint8)
            out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(out_gray, 0, 1, cv2.THRESH_BINARY)        
            mask = cv2.merge((mask, mask, mask))
            score = similarityScore(out, dst_cropped, mask)
            print("Similarity Score: {}".format(score))

            cv2.imwrite("oflow.jpg",out)
            out = out*0.5+dst_cropped*0.5
            cv2.imwrite("piecewise.jpg",out)

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
    matched_img = cv2.drawMatches(src_image,kp1,dst_image,kp2,good,None,**draw_params)
    cv2.imwrite("matches.jpg",matched_img)
            
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
