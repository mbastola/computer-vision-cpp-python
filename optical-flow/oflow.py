"""
Manil Bastola

Testing Optical flow algorithms in motion tracking of moving bodies in a video stream

This code is heavily based on Opencv 3.1 Optical Flow example. 

"""

import sys
import numpy as np
import cv2 as cv
from datetime import datetime

def uniqueStr():
    nowinsec = int((datetime.now()-datetime(1970,1,1)).total_seconds())
    return str(nowinsec)    

def denseFlow(path):
    cap = cv.VideoCapture(path or 0)
    ret, frame0 = cap.read()
    if not ret:
        print("Cannot read video input")
        return 
    #set prev as greyscale
    prev = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame0)
    hsv[...,1]=255
    
    j = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #set current grayscale
        cur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #Uses Gunnar Farneback for dense oflow
        flow = cv.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, angle = cv.cartToPolar(flow[...,0], flow[...,1])
        #angle in degrees
        hsv[...,0] = angle * (180/(np.pi/2))
        #normalize the magnitude
        hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        final = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        
        unq = uniqueStr()
        output = "output/oflow_{}.png".format(unq)
        output_in = "output/in_{}.png".format(unq)
        #output = "output/oflow_{}.png".format(j)
        cv.imwrite(output, final)
        cv.imwrite(output_in, frame)
        j+=1


def sparseFlow(path):
    ft_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    cap = cv.VideoCapture(path or 0)
    ret, frame0 = cap.read()
    if not ret:
        print("Cannot read video input")
        return 
    #set prev as greyscale
    prev = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
    prev_feats = cv.goodFeaturesToTrack(prev, mask=None, **ft_params)
    color = np.random.randint(0,255,(100,3))
    mask = np.zeros_like(frame0)
    
    j = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #set current grayscale
        cur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #implements lucas-kanades oflow
        feats, status, errs = cv.calcOpticalFlowPyrLK(prev, cur, prev_feats, None, **lk_params)

        # Select good points
        good_new = feats[status==1]
        good_old = prev_feats[status==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        final = cv.add(frame,mask)
        
        unq = uniqueStr()
        #output = "output/6/oflow_{}.png".format(unq)
        #output_in = "output/in_{}.png".format(unq)
        if j%5 == 0:
            output = "output/oflow_{0:0=3d}.png".format(j)
            cv.imwrite(output, final)
        #cv.imwrite(output_in, frame)
        j+=1
        # update the previous frame and previous points
        prev = cur.copy()
        prev_feats = good_new.reshape(-1,1,2)
        
    
def main():
    path = None
    if len(sys.argv) > 1:
        path = sys.argv[1];

    sparseFlow(path)
    #denseFlow(path)

if __name__ == "__main__":
    main()
        
