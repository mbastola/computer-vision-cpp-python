""""
Manil Bastola

The Coloring portion is based on: http://richzhang.github.io/colorization
"""

import cv2 as cv
import numpy as np
import sys

def colorizeImg(img):
    # Select caffe model
    net = cv.dnn.readNetFromCaffe("colorization_deploy_v2.prototxt","colorization_release_v2.caffemodel")

    # load cluster centers
    pts_in_hull = np.load('pts_in_hull.npy') 

    # populate cluster centers as 1x1 convolution kernel
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    img_rgb = (img[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_BGR2LAB)
        
    # pull out L channel
    img_l = img_lab[:,:,0]
    # get original image size
    (H_orig,W_orig) = img_rgb.shape[:2] 
        
    # resize image to network input size
    img_rs = cv.resize(img_rgb, (224, 224)) 
        
    # resize image to network input size
    img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
    img_l_rs = img_lab_rs[:,:,0]
    
    # subtract 50 for mean-centering
    img_l_rs -= 50 
    
    net.setInput(cv.dnn.blobFromImage(img_l_rs))
    
    # this is our result
    ab_dec = net.forward('class8_ab')[0,:,:,:].transpose((1,2,0)) 
    
    (H_out,W_out) = ab_dec.shape[:2]
    ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) 
        
    # concatenate with original image L
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)
    
    # show original image
    #cv.imshow('Original', img)
    # Resize the corlized image to it's orginal dimensions 
    img_bgr_out = 255*cv.resize(img_bgr_out, (W_orig, H_orig), interpolation = cv.INTER_AREA)
    cv.imwrite('ColorizedRestored.jpg', img_bgr_out)
    return img_bgr_out.astype('uint8')

def contrast(img):
    alpha = 2.0
    beta = -200

    new = alpha * img + beta
    new = np.clip(new, 0, 255).astype(np.uint8)
    return new
    
def genDamageMask(in_img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv.filter2D(in_img, -1, kernel)
    method = 1
    if method == 0:
        #img = contrast(img)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(img_gray, 245, 255, cv.THRESH_BINARY)

        #needs tweaking with the kernels
        mask = cv.bilateralFilter(mask,9,75,75)
        cv.imwrite("tmp3.jpg",mask)         
        kernel = np.ones((2,2),np.uint8)
        mask = cv.erode(mask,kernel,iterations = 1)
        kernel = np.ones((12,12),np.uint8)
        mask = cv.dilate(mask,kernel,iterations = 2)
        cv.imwrite("tmp2.jpg",mask) 
        return mask
    else:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(img_gray, 220, 255, cv.THRESH_BINARY)
        
        #more robust to noise
        # wherever it is marked white (possible foreground), change mask=2
        # wherever it is marked black (sure background), change mask=0
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        mask[mask != 0] = 3
        mask[mask == 0] = 2
        mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        cv.imwrite("tmp3.jpg",255*mask)         
        kernel = np.ones((8,8),np.uint8)
        mask = cv.dilate(mask,kernel,iterations = 2)
        cv.imwrite("tmp2.jpg",255*mask)         
        return mask
    

def restoreImg(in_img, mask = None):
    if not mask:
        mask = genDamageMask(in_img)
    restored = cv.inpaint(in_img, mask, 3 ,cv.INPAINT_TELEA)
    cv.imwrite("restored.jpg",restored) 
    return restored

def main():
    infile = None
    maskfile = None
    restore = True
    colorize = False
    
    if len(sys.argv) == 3:
        infile = sys.argv[1]
        maskfile = sys.argv[2]
    elif len(sys.argv) == 2:
        infile = sys.argv[1]
    else:
        print("Input file name needed")
        return
    
    in_img = cv.imread(infile, cv.IMREAD_COLOR)
    rows,cols,ch = in_img.shape

    if rows < 960:
        orows = 960;
        ocols = int(orows/rows * cols)
        in_img = cv.resize(in_img,(ocols,orows))
    
    in_mask = None
    if maskfile:
        in_mask = cv.imread(maskfile)
        in_mask = cv.resize(in_img,(ocols,orows))
    
    img = None
    if ch == 1:
        colorize = True
        img = in_img
    else:
        img = in_img
        #img = cv.cvtColor(in_img, cv.COLOR_BGR2GRAY)
    #cv.imwrite("tmp.jpg",img)
    if restore:
        img = restoreImg(img, in_mask)
    if colorize:
        img  =  colorizeImg(img)
    
if __name__ == '__main__':
    main()
