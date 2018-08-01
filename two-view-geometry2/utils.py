import os
import sys
import cv2
import _pickle as pickle
import math
import numpy as np


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
    sift = cv2.xfeatures2d.SIFT_create()

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


def unitDeterminantCheck(M):
    return abs(np.linalg.det(M)) - 1.0 < 1e-7

def decompEssentialMatrix(E):
    # svd::modify_A
    _, u, vt = cv2.SVDecomp(E, 1)
    if np.linalg.det(vt) < 0:
        vt *= -1
    if np.linalg.det(u) < 0:
        u *= -1
    """R = svd.u * Mat(W) * svd.vt;"""
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]]);
    R1 = np.matmul( u, np.matmul( W, vt ) )
    R2 = np.matmul( u, np.matmul( W.T, vt ) )
    t = u[:,-1]
    if not unitDeterminantCheck(R1) and not unitDeterminantCheck(R2):
        print("Not rotation matrices")
    return R1, R2, t
