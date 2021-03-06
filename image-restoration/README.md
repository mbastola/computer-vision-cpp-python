[This project](https://github.com/mbastola/computer-vision-cpp-python/tree/master/image-restoration) tackles image restoration and coloring for old damaged black and white images. The restoration problem tackels automatic damaged region detection which turns out to be a hard problem to automate. The successful resotrations still require some form of tweaking the Kernel parameters (dilate/erode/color threshold, etc.) in the code. The coloring portion Opencv DNN based Caffe framework to colorize black and white images. One can find the Caffe models [here](https://github.com/richzhang/colorization/tree/master/models)

Restoration is based on two methods which can be used depending on the situation. First uses simple thresholding + bilateral filter to generate the masks. This is useful in cases where images need not be context aware and the foreground/background separation is good. The second method uses thesholding followed by foreground/background segmentation which is robust in cases where background and foreground color are in close range. Both methods currently only work for whitened damaged regions but can be extended to any colored regions. One can note the difference in masks for first method and second method below:


<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/mask_0.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/mask_1.jpg" width="320" /> 
</p>

```
Damage masks generated by method 1 and 2 respectively
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/temp_0.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/temp_1.jpg" width="320" /> 
</p>

```
Corresponding outputs from the damage masks. Note that method 2 performs better in this example.
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/mask_2.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/mask_3.jpg" width="320" /> 
</p>

```
Damage masks generated by method 1 and 2 respectively
```

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/restored_8.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/temp_2.jpg" width="320" /> 
</p>

```
Corresponding outputs from the damage masks. Note that method 1 performs better in this example.
```


## Some samples:


### Restoration Only (Watermarks removal)

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/8.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/restored_8.jpg" width="320" /> 
</p>

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/9.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/restored_9.jpg" width="320" /> 
</p>

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/10.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/restored_10.jpg" width="320" /> 
</p>

### Colorize Only

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/11.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/colorized_11.jpg" width="320" /> 
</p>

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/12.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/colorized_12.jpg" width="320" /> 
</p>

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/13.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/colorized_13.jpg" width="320" /> 
</p>


### Restoration + Coloring

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/1.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/restored_1.jpg" width="320" /> 
</p>

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/2.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/restored_2.jpg" width="320" /> 
</p>

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/4.png" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/restored_4.jpg" width="320" /> 
</p>

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/5.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/restored_5.jpg" width="320" /> 
</p>

<p float="left">
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/7.jpg" width="320" />
  <img src="https://github.com/mbastola/computer-vision-cpp-python/blob/master/image-restoration/imgs/restored_7.jpg" width="320" /> 
</p>




