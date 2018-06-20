In [this project](https://github.com/mbastola/computer-vision-cpp-python/tree/master/camera-calibration), I use simple manual camera calibation method. The setup is as follows:

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/camera-calibration/imgs/img1.jpeg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/camera-calibration/imgs/img2.jpeg" width="320" />
</p>


Assuming pin hole camera, we can utilizing similar triangle methods(see below), and can come up with focal lengths equation: 
    # fx = (dx/dX) * dZ , fy = (dy/dY) * dZ
where dX, dY are the physical length & width of the object in view and dZ is the distance from object to camera. dx and dy are the corresponding pixel width & height of the object

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/camera-calibration/imgs/pinhole.png" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/camera-calibration/imgs/calib.jpeg" width="320" />
</p>


```python
import numpy as np
```


```python
#measure object width,height,distance in inches
obj_width = 18.75
obj_height = 13.5
obj_distance = 35.2
```


```python
#get object width,height in pixels
obj_img_width = 1784
obj_img_height = 1268
```


```python
fx
```




    3349.1626666666666




```python
# fx = (dx/dX) * dZ , fy = (dy/dY) * dZ
fx = ( obj_img_width / obj_width ) * obj_distance 
fy = ( obj_img_height / obj_height ) * obj_distance 
```


```python
print(fx, fy)
```

    3349.1626666666666 3306.192592592593


Here we write a generic function that takes in the above params and gives us the camera matrix


```python
# img_size if there has been any rescaling of the original image
def simple_calibration(img_size, full_img_size, focal_lengths ):
    row,col = img_size
    full_row, full_col = full_img_size
    fx,fy = focal_lengths
    fx = fx*col/full_col
    fy = fy*row/full_row
    K = np.diag([fx,fy,1])
    K[0,2] = 0.5*col
    K[1,2] = 0.5*row
    return K
```


```python
K = simple_calibration((4032,3024), (4032,3024), (fx,fy) )
```


```python
K
```




    array([[3.34916267e+03, 0.00000000e+00, 1.51200000e+03],
           [0.00000000e+00, 3.30619259e+03, 2.01600000e+03],
           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


