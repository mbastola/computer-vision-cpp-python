# Feature Matching

This project implements feature generation and matching using OpenCV library. We observe two classes of object types: The first is solid cube shaped object such as a cereal box that is ideal for homography transforms and post processes. The second is deformable object such as a bag of chips that can be physically distored between two images and homographies and post processing may fail on them. In this project we observe the usage of optical flow on the 2nd category to check viability of estimating such distorions. 

### Feature generation with SIFT
<p float="left">
  <img src="imgs/kraft_orig.png" width="150" />
  <img src="imgs/kraft_orig_feats.kp_kps.png" width="150" />
  <img src="imgs/mac-cheeze-table.png" width="250" />	
  <img src="imgs/mac-cheeze-table_feats.kp_kps.png" width="250" />	
</p>

<p float="left">
  <img src="imgs/doritos.jpg" width="150" />
  <img src="imgs/doritos_feats.kp_kps.png" width="150" />
  <img src="imgs/doritos_store.jpg" width="250" />	
  <img src="imgs/doritos_store_feats.kp_kps.png" width="250" />	
</p>

### Feature matching with FLANN based KNN Matcher
<p float="left">
  <img src="imgs/matches.jpg" width="800" />
</p>
<p float="left">
  <img src="imgs/matches2.jpg" width="800" />
</p>


### Homography estimation and Transform
<p float="left">
  <img src="imgs/transform.jpg" width="800" />
</p>
<p float="left">
  <img src="imgs/homography_transform2.jpg" width="800" />
</p>

Below shows the source and the destination images rescaled to destination sizes with source images transformed to the perspectives of the destination image. The third image shows difference in pixels between the two. Note large difference in the 2nd category item.
<p float="left">
  <img src="imgs/src_transformed1.jpg" width="250" />
  <img src="imgs/dst_orig1.jpg" width="250" />
  <img src="imgs/homography_difference1.jpg" width="250" />

<p float="left">
  <img src="imgs/src_transformed.jpg" width="250" />
  <img src="imgs/dst_orig.jpg" width="250" />
  <img src="imgs/homography_difference.jpg" width="250" />
</p>

### Optical Flow to visualize disortions

The usage of Optical flow provides vectors to the deformation in the destination object with respect to source.
<p float="left">
  <img src="imgs/tmp2.jpg" width="400" />
  <img src="imgs/tmp1.jpg" width="400" />	
</p>

<p float="left">
  <img src="imgs/tmp4.jpg" width="400" />
  <img src="imgs/tmp3.jpg" width="400" />	
</p>
    