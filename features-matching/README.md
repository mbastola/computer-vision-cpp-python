
[This project](https://github.com/mbastola/computer-vision-cpp-python/tree/master/features-matching) implements feature generation and matching using OpenCV library. We observe two classes of object types: The first is solid cube shaped object such as a cereal box that is ideal for homography transforms and post processes. The second is deformable object such as a bag of chips that can be physically distored between two images and homographies and post processing may fail on them. In this project we observe the usage of optical flow on the 2nd category to check viability of estimating such distortions. I find that the optical flow optimization methodpost homography is robust to deformable objects and provides more complex mappings from source to destination images which had been a significant limitation of homographies alone.


### Feature generation with SIFT
<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/kraft_orig.png" width="150" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/kraft_orig_feats.kp_kps.png" width="150" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/mac-cheeze-table.png" width="250" />	
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/mac-cheeze-table_feats.kp_kps.png" width="250" />	
</p>

```
Item category 1: Solid Box
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/doritos.jpg" width="150" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/doritos_feats.kp_kps.png" width="150" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/doritos_store.jpg" width="250" />	
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/doritos_store_feats.kp_kps.png" width="250" />	
</p>

```
Item category 2: Deformable Bags
```


### Feature matching with FLANN based KNN Matcher
<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/matches.jpg" width="800" />
</p>
<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/matches2.jpg" width="800" />
</p>


### Homography estimation and Transform
<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/transform.jpg" width="800" />
</p>
<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/homography_transform1.jpg" width="800" />
</p>

Below shows the source and the destination images rescaled to destination sizes with source images transformed to the perspectives of the destination image. The third image shows difference in pixels between the two. Note large difference in the 2nd category item.

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/src_transformed1.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/dst_overlay1.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/homography_difference1.jpg" width="250" />


```
Normalized Correlation Score: 0.977
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/src_transformed.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/dst_overlay.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/homography_difference.jpg" width="250" />
</p>

```
Normalized Correlation Score: 0.742
```

The normalized correlation score shows us the similarity between source and destination images when overlaid on top of each other. Note that the score is way higher 0.977 for category 1 item vs category 2 item with score of 0.742 only. This difference can also be observed with the difference image shown in the third column above.


### Optical Flow to visualize disortions

The usage of Optical flow provides vectors to the deformation in the destination object with respect to source. Note minimal flow in the first category, vs large flow in the second category as expected.


<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/tmp2.jpg" width="400" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/tmp1.jpg" width="400" />	
</p>

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/tmp4.jpg" width="400" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/tmp3.jpg" width="400" />	
</p>

Using optical flow, we track the motion of each feature point from source to the destination image. We then remap the source image to destination using SKImages Piecewise Affine transform function. The result is a significant increase in the correlation score! 

In the images below, left is source image transformed post optical flow. The middle shows alignment when overlayed. The third shows difference in pixels

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/oflow1.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/piecewise1.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/difference1.jpg" width="250" />
</p>

```
Normalized Correlation Score is 0.982
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/oflow.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/piecewise.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/difference.jpg" width="250" />
</p>

```
Normalized Correlation Score has increased to 0.806 from 0.742
```

### Another Item of Category 2


<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/large_bag.jpg" width="400" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/large_bag_scene.jpg" width="400" />
</p>

```
Source and Destination images.
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/homography_transform2.jpg" width="800" />
</p>


<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/src_transformed2.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/dst_overlay2.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/homography_difference2.jpg" width="250" />
</p>

```
Homography Transform. Normalized Correlation Score is 0.892
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/tmp6.jpg" width="400" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/tmp5.jpg" width="400" />	
</p>

```
Optical flow vectors
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/oflow2.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/piecewise2.jpg" width="250" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/features-matching/imgs/difference2.jpg" width="250" />
</p>

```
Optical Flow based Transform. Normalized Correlation Score has increase to 0.93
```

