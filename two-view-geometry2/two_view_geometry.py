#!/usr/bin/env python
#Manil Bastola

import os
import sys
import cv2
import _pickle as pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

        
def plot3D(pts3d):
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    ax.scatter3D(pts3d[:,0],pts3d[:,1],pts3d[:,2])


def getNormalizedCoordinates(a, kinv):
    #converts 2x1 array  to normalized (3x1) coordinates
    a_hg = np.array([a[0],a[1],1])
    a_calib = np.matmul(kinv, a_hg)
    return a_calib
    

def getVectorCrossMatrix(a):
    #converts a 3x1 vector to 3x3 cross product matrix (Skew-symmetric)
    a_cross = np.zeros((3,3))
    a_cross[0,1] = -a[2]
    a_cross[0,2] = a[1]
    a_cross[1,0] = a[2]
    a_cross[1,2] = -a[0]
    a_cross[2,0] = -a[1]
    a_cross[2,1] = a[0]
    return a_cross


#https://perception.inrialpes.fr/Publications/1997/HS97/HartleySturm-cviu97.pdf
def LinearLSTriangulation( u, P, u1, P1, mode=0):
    u_cross = getVectorCrossMatrix(u)
    u1_cross = getVectorCrossMatrix(u1)
    A0 = np.matmul(u_cross,P)
    A1 = np.matmul( u1_cross, P1 )
    A_full = np.concatenate([A0, A1])

    A = A_full[:,:-1]
    B = -A_full[:,-1]
    if mode == 0:
        X = np.linalg.lstsq(A,B,rcond=-1)
        X = np.array( [ X[0][0], X[0][1], X[0][2], 1 ] )
    else:
        #try cv2 lstsq solver using SVD Decomp
        X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
        X = np.array( [ X[1][0][0], X[1][1][0], X[1][2][0], 1 ] )
    return X

    
def get3DReprojectionError(X, K, P, x):
    x_uv = np.matmul( np.matmul( K, P) , X) 
    x_uv_inhg = np.array( [x_uv[0]/x_uv[2], x_uv[1]/x_uv[2] ] )
    return np.linalg.norm(x_uv_inhg - x)
    
def TriangulatePoints( ptsa, ptsb, K, Kinv, Pa, Pb ):
    ptcloud = []
    reprojection_errs = []
    for a,b in zip(ptsa, ptsb):
        ua = getNormalizedCoordinates(  a , Kinv )
        ub = getNormalizedCoordinates( b , Kinv )
        X = LinearLSTriangulation( ua, Pa, ub, Pb )
        err = get3DReprojectionError(X, K, Pb, b )
        ptcloud.append(X)
        reprojection_errs.append(err)
    err = np.array(reprojection_errs).mean()
    return np.array(ptcloud), err

def two_view_geometry( pts1, pts2, F, K ):
    Kinv = np.linalg.inv(K)
    """Pose for 1st view"""
    P0 = np.eye(3,4)
    
    """"E = K.T * F * K"""
    E = np.matmul( K.T , np.matmul( F, K ) )
    R1, R2, t = decompEssentialMatrix(E)
    
    """Poses for 2nd view. Four possilbe solutions based on chirality"""
    
    P1 = np.column_stack( [ R1, t ] )
    P2 = np.column_stack( [ R2, t ] )
    P3 = np.column_stack( [ R1, -t ] )
    P4 = np.column_stack( [ R2, -t ] )
    
    Ps = [P1,P2,P3,P4]
    errs = []
    pts3ds = []
    
    """Pick the projection that minimizes the reprojection error"""
    for P in Ps:
        pts3d, err =  TriangulatePoints( pts1, pts2, K, Kinv, P0, P )
        errs.append(err)
        pts3ds.append(pts3d)
    
    min_err_idx = np.argmin( np.array(errs) )
    print("Min Reprojection error: {}".format( errs[min_err_idx] )) 
    return pts3ds[ min_err_idx ]


# In[255]:


def main():
    src_image_path1 = sys.argv[1]
    src_image_path2 = sys.argv[2]
    calibration_mat_path = sys.argv[3]
    distortions_vec_path = sys.argv[4] if len(sys.argv) == 5 else None
    debug = False
    
    src_image_1 = cv2.imread(src_image_path1, cv2.IMREAD_COLOR)
    src_image_2 = cv2.imread(src_image_path2, cv2.IMREAD_COLOR)
    
    K = np.loadtxt(calibration_mat_path)
    if distortions_vec_path:
        distortions = np.loadtxt(distortions_vec_path)
        src_image_1 = cv2.undistort(src_image_1, K, distortions, None, None)
        src_image_2 = cv2.undistort(src_image_2, K, distortions, None, None)
        
    src_gray_1 = cv2.cvtColor(src_image_1 , cv2.COLOR_BGR2GRAY)
    src_gray_2 = cv2.cvtColor(src_image_2 , cv2.COLOR_BGR2GRAY)
    
    kp1, desc1 = computeDescriptions(src_gray_1, "feats/{}_feats.kp".format(os.path.basename(src_image_path1).split(".")[0]))
    kp2, desc2 = computeDescriptions(src_gray_2, "feats/{}_feats.kp".format(os.path.basename(src_image_path2).split(".")[0]))

    matcher = genMatcher()
    knn_matches = matcher.knnMatch(desc1, desc2, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.8
    good = []
    pts1 = []
    pts2 = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt) 

    if len(good) <= 10 : #MIN_MATCH_COUNT=10
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        return
        
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC )  #FM_LMEDS

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    points_3d = two_view_geometry( pts1, pts2, F, K )
    out_pts = np.array([ [pts[0],pts[1],pts[2]] for pts in points_3d ])
    out_colors = np.array([ src_image_1[pt[1],pt[0]] for pt in pts1  ])
    
    if debug:
        plot3D(pts3d)
    
    write_ply("./out1.ply", out_pts, out_colors)
    
    
if __name__ == "__main__":
    main()
