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
from scipy.interpolate import griddata
#from skimage.transform import PiecewiseAffineTransform, PolynomialTransform, warp

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
        return unpickle_keypoints(featsfilename)
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(0,3,0.07)

    #compute keypoint & descriptions
    kp1, des1 = sift.detectAndCompute(img,None)
    if featsfilename:
        temp_array = []
        temp = pickle_keypoints(kp1, des1)
        temp_array.append(temp)
        pickle.dump(temp_array, open(featsfilename, "wb"))
        out=cv2.drawKeypoints(img, kp1,  0, color=(0,255,0), flags=0)
        cv2.imwrite( featsfilename+"_kps.png", out);
    return kp1, des1


def genMatcher():
    #use FLANN_INDEX_KDTREE algorithm
    flann_params = dict(algorithm = 1, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    return matcher

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
            score = cv2.matchTemplate(src_transformed_cropped, dst_cropped,cv2.TM_CCOEFF_NORMED)

            print("Similarity Score: {}".format(score))

            cv2.imwrite("src_transfromed.jpg",src_transformed_cropped)
            cv2.imwrite("dst_orig.jpg",dst_cropped)

            diff = cv2.absdiff(dst_cropped,src_transformed_cropped)
            #score2 = np.mean(diff)/255
            #print(score2)
            cv2.imwrite("homography_difference.jpg",diff)
            
            dst_image = cv2.polylines(dst_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            
            
            #weighted blend for vizualization
            src_transformed = 0.5*src_transformed+0.5*dst_image
            cv2.imwrite("homography_transform.jpg",src_transformed)

        if OPTICAL_FLOW:

            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (100,100),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS |cv2.TERM_CRITERIA_COUNT, 30, 0.05))
            
            # params for ShiTomasi corner detection
            feature_params = dict( maxCorners = 2000,
                                   qualityLevel = 0.05,
                                   minDistance = 7,
                                   blockSize = 7)

            src_transformed_cropped_gray = cv2.cvtColor(src_transformed_cropped , cv2.COLOR_BGR2GRAY)
            dst_cropped_gray = cv2.cvtColor(dst_cropped , cv2.COLOR_BGR2GRAY)
            color = np.random.randint(0,255,(100,3))
            p0 = cv2.goodFeaturesToTrack(src_transformed_cropped_gray, mask = None, **feature_params)
            #_, p0 = computeDescriptions(src_transformed_cropped)
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(src_transformed_cropped, dst_cropped, p0, None, **lk_params)
            # Select good points
            good_old = p0[st==1]
            good_new = p1[st==1]

            for i,(new,old) in enumerate(zip(good_new,good_old)):
                try:
                    a,b = new.ravel()
                    c,d = old.ravel()
                    src = cv2.line(src_transformed_cropped, (a,b),(c,d), color[i].tolist(), 2)
                    dst = cv2.circle(dst_cropped,(a,b),5,color[i].tolist(),-1)
                except:
                    continue
                
            cv2.imwrite("tmp3.jpg",src);
            cv2.imwrite("tmp4.jpg",dst);

            """
            h,w,_ = dst_cropped.shape

            grid_x, grid_y = np.mgrid[0:h-1:h*1j, 0:w-1:w*1j]
            #grid_z = griddata(dst_pts.reshape(-1,2), src_pts.reshape(-1,2), (grid_x, grid_y), method='cubic')
            grid_z = griddata(good_old, good_new, (grid_x, grid_y), method='cubic')
            map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(h,w)
            map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(h,w)
            map_x_32 = map_x.astype('float32')
            map_y_32 = map_y.astype('float32')
            src_transformed_oflow = cv2.remap(src_transformed_cropped_gray, map_x_32 , map_y_32 , cv2.INTER_CUBIC)
            
            cv2.imwrite("oflow_transform.jpg", src_transformed_oflow );
            """
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
    matched_img = cv2.drawMatches(src_image,kp1,dst_image,kp2,good,None,**draw_params)
    cv2.imwrite("matches.jpg",matched_img)
            
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
