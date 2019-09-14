In [this project]([https://github.com/mbastola/computer-vision-cpp-python/tree/master/multi-view-geometry), I extend my two-view geometry project to n-view scenario. This is the first of the series where we deal with computing multi view pnp pose and sparse 3D reconstruction.


```python
import os
import sys
import cv2
import _pickle as pickle
#import cPickle as pickle
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
from two_view_geometry import two_view_geometry,TriangulatePoints, unitDeterminantCheck,genMatcher, computeDescriptions, plot3D, write_ply
```

Lets see what images we have for this reconstruction:


```python
#https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/modelFountain.html
ff, axs = plt.subplots(2,4,figsize=(10,5))
idx=0
for fname in glob.glob('./imgs/fountain/*.png')[:8]:
    img = cv2.imread(fname)
    plt.subplot(int("24{}".format(idx+1)))
    plt.imshow(img.T[[2,1,0]].T)
    idx+=1
plt.tight_layout()
plt.show()
```

![png](https://github.com/mbastola/computer-vision-cpp-python/blob/master/multi_view_structure_sfm/imgs/output_5_0.png)






```python
images_dir = "./imgs/fountain/"
filenames = os.listdir(images_dir)
filenames.sort()
```

Lets begin with 2 view case for first set of camera extrinsics


```python
src_image_path1 = "{}{}".format(images_dir,filenames[0])
src_image_path2 = "{}{}".format(images_dir,filenames[1])

```


```python
calibration_mat_path = "./calib.txt"
distortions_vec_path = None
debug = False

```

Same code from previous [two view project](https://github.com/mbastola/computer-vision-cpp-python/tree/master/two-view-geometry2):


```python

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

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC )  #FM_LMEDS

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

Kinv = np.linalg.inv(K)

points_3d, P0, P1 = two_view_geometry( pts1, pts2, F, K )
```

    Min Reprojection error: 0.6652368589262047



```python
print(P1)
```




    array([[ 0.98754241, -0.02270042, -0.15570704,  0.99689069],
           [ 0.02604214,  0.99947151,  0.01945509,  0.00868799],
           [ 0.15518311, -0.02326767,  0.98761167, -0.07831654]])



Now for each additonal image we estimate the new Projectiion matrix( Rotation & translation ) based on the previous result. This is accomplished by using opencv's solve p&p algorithm. Meanwhile we also do some bookkeeping for the 3d-2d correspondences for 2 reasons: 1) we only add new 3d points if it wasnt seen in previous view, 2) We build foundation for global bundle adjustment optimization. 


```python
kp_prev = kp2
desc_prev = desc2
src_image_prev = src_image_2
pts_prev = pts2
points_3d_prev = points_3d
pose_prev = P1

out_pts_3d = points_3d.copy()
out_colors = np.array([ src_image_1[pt[1],pt[0]] for pt in pts1 ])

#for bundle adjustment
out_pts_2d = pts_prev.copy()
rvec,_ = cv2.Rodrigues(P1[:,:3])
out_camera_params = np.hstack([rvec.ravel(), P1[:,3].ravel() ])

view_idx = 0
points_indexes = np.arange(points_3d_prev.shape[0] )
points_indexes_prev = points_indexes.copy()

camera_indexes = np.full( (pts2.shape[0],1), view_idx )

for fname in filenames[2:]:
    src_image_path = "{}{}".format(images_dir,fname)
    print( src_image_path )
    src_image = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    src_gray = cv2.cvtColor(src_image , cv2.COLOR_BGR2GRAY)
    kp, desc = computeDescriptions(src_gray, "feats/{}_feats.kp".format(os.path.basename(src_image_path).split(".")[0]))
    
    pts_prev_keys = [ ",".join( list(pt.astype(str) ) ) for pt in pts_prev ]
    points_3d_2d = dict( zip( pts_prev_keys, points_indexes_prev ))
    
    knn_matches = matcher.knnMatch(desc_prev, desc, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.8
    good = []
    pts_a = []
    pts_b = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
            pts_b.append(kp[m.trainIdx].pt)
            pts_a.append(kp_prev[m.queryIdx].pt) 
    
    pts_a = np.array(pts_a)
    pts_b = np.array(pts_b)
    
    # Using Epiplolar Constraint to filter only inlier points
    F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC )
    pts_a = pts_a[mask.ravel()==1]
    pts_b = pts_b[mask.ravel()==1]
    
    pts_a = np.int32(pts_a)
    
    
    #Select 3d points that were seen in previous image view
    points_3d_cur_index = np.array([ points_3d_2d.get( ",".join( list(pt.astype(str)) ) ) for pt in pts_a ])
    pts_mask = np.array([ True if idx != None else False for idx in points_3d_cur_index ])
    points_3d_cur = np.array([ out_pts_3d[idx] if idx != None else [0,0,0,1] for idx in points_3d_cur_index])
    
    retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(points_3d_cur[ pts_mask ][:,:-1], pts_b[pts_mask] , K, None)
    
    pts_b = np.int32(pts_b)
    
    R_mat, _ = cv2.Rodrigues(rvecs)
    if not unitDeterminantCheck(R_mat):
        print("Not a rotation matrix")
        break
    
    #triangulate new 3d points
    pose = np.column_stack( [ R_mat, tvecs ] )
    points_3d_cur2, err =  TriangulatePoints( pts_a[ ~pts_mask ], pts_b[ ~pts_mask ], K, Kinv, pose_prev, pose )

    
    print("Reprojection error:", err)
    
    points_3d_cur[~pts_mask] = points_3d_cur2
    
    
    #for bundle adjustment
    out_camera_params = np.vstack( [ out_camera_params, np.hstack([rvecs.ravel(), tvecs.ravel() ]) ])
    out_pts_2d = np.concatenate( [ out_pts_2d, pts_b ], axis=0)
    #points_indexes_cur = np.concatenate([ points_3d_cur_index[ pts_mask ], out_pts_3d.shape[0] + np.arange(points_3d_cur2.shape[0]) ])  
    
    points_indexes_cur = points_3d_cur_index.copy()
    points_indexes_cur[ ~pts_mask ] = out_pts_3d.shape[0] + np.arange(points_3d_cur2.shape[0])   
    
    points_indexes = np.concatenate( [ points_indexes, points_indexes_cur ], axis=0)
   
     
    out_pts_3d = np.concatenate([out_pts_3d, points_3d_cur2 ], axis=0)
    #print(out_pts_3d.shape, points_3d_cur.shape)
    out_colors_cur = np.array([ src_image_prev[pt[1],pt[0]] for pt in pts_a[ ~pts_mask ]  ])
    out_colors = np.concatenate([out_colors, out_colors_cur],axis=0)
    camera_indexes = np.concatenate( [ camera_indexes, np.full( (pts_b.shape[0],1), view_idx+1 ) ] )
    
    
    
    kp_prev = kp
    desc_prev = desc
    src_image_prev = src_image
    pts_prev = pts_b
    points_indexes_prev = points_indexes_cur
    pose_prev = pose
    view_idx += 1 
    
```

    ./imgs/fountain/0002.png
    Reprojection error: 0.4461336215020665
    ./imgs/fountain/0003.png
    Reprojection error: 0.6728600949007068
    ./imgs/fountain/0004.png
    Reprojection error: 0.9330157178213542
    ./imgs/fountain/0005.png
    Reprojection error: 1.0826329659375762
    ./imgs/fountain/0006.png
    Reprojection error: 0.8006911486304148
    ./imgs/fountain/0007.png
    Reprojection error: 0.5700519795581112
    ./imgs/fountain/0008.png
    Reprojection error: 0.7531404114916702
    ./imgs/fountain/0009.png
    Reprojection error: 0.6721930930126719
    ./imgs/fountain/0010.png
    Reprojection error: 0.6366139774133439


Cool, really low reprojection errors. Lets visualize the output 3d points:


```python
plot3D(out_pts_3d)
```


    
![png](https://github.com/mbastola/computer-vision-cpp-python/blob/master/multi_view_structure_sfm/imgs/output_13_0.png)
    


Okay, matplotlib is limited. So I will export the geometry and open it in meshlab


```python
#note opencv is bgr space vs rgg
write_ply("./out.ply", out_pts_3d[:,:-1], out_colors.T[[2,1,0]].T)
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/multi_view_structure_sfm/imgs/out.gif" width="900"/>
</p>

![gif](https://github.com/mbastola/computer-vision-cpp-python/blob/master/multi_view_structure_sfm/imgs/out.gif)

## References


Multiple View Geometry in Computer Vision (second edition), R.I. Hartley and A. Zisserman, Cambridge University Press, ISBN 0-521-54051-8

