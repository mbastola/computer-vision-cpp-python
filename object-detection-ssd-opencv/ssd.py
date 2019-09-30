# This script is used to demonstrate MobileNet-SSD network using OpenCV deep learning module.
#
# It works with model taken from https://github.com/chuanqi305/MobileNet-SSD/ that
# was trained in Caffe-SSD framework, https://github.com/weiliu89/caffe/tree/ssd.
# Model detects objects from 20 classes.
#
# Also TensorFlow model from TensorFlow object detection model zoo may be used to
# detect objects from 90 classes:
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# Text graph definition must be taken from opencv_extra:
# https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt

import sys
import numpy as np
import json
import cv2 as cv
from datetime import datetime

#0 for 20 class Caffe model, 1 for 90 class Tensorflow Model
MODEL = 0

def readJson(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data


def uniqueStr():
    nowinsec = int((datetime.now()-datetime(1970,1,1)).total_seconds())
    return str(nowinsec)
        
def CaffeNet():
    Model = {}
    Model["idx"] = 0
    Model["model"] = cv.dnn.readNetFromCaffe("caffe/MobileNetSSD_deploy.prototxt", "caffe/MobileNetSSD_deploy.caffemodel")
    Model["classes"] = readJson("caffe/classes.json")
    Model["numClasses"] = len(Model["classes"])
    Model["conf_thresh"] = 0.2
    return Model

def TfNet():
    Model = {}
    Model["idx"] = 1
    Model["model"] = cv.dnn.readNetFromTensorflow("tensorflow/frozen_inference_graph.pb", "tensorflow/ssd_mobilenet_v1_coco.pbtxt")
    Model["classes"] = readJson("tensorflow/classes.json")
    Model["numClasses"] = len(Model["classes"])
    Model["conf_thresh"] = 0.2
    return Model

def main():
    videopath = None
    if len(sys.argv) > 1:
        videopath = sys.argv[1];
    Model = None
    if MODEL == 0:
        Model = CaffeNet()
    else:
        Model = TfNet()

    cap = cv.VideoCapture(videopath or 0)
    

    inWidth = 300
    inScaleFactor = 0.007843
    meanVal = 127.5

    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH )
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT )
        WHRatio = width / float(height)
        
        inHeight = int(inWidth*WHRatio)

        blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), bool(Model["idx"]))
        Model["model"].setInput(blob)
        detections = Model["model"].forward()

        cols = frame.shape[1]
        rows = frame.shape[0]

        if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = y1 + cropSize[1]
        x1 = int((cols - cropSize[0]) / 2)
        x2 = x1 + cropSize[0]
        frame = frame[y1:y2, x1:x2]

        cols = frame.shape[1]
        rows = frame.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > Model["conf_thresh"]:
                class_id = int(detections[0, 0, i, 1])

                #get bounding boxes
                x0 = int(detections[0, 0, i, 3] * cols)
                y0 = int(detections[0, 0, i, 4] * rows)
                x1   = int(detections[0, 0, i, 5] * cols)
                y1   = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (x0, y0), (x1, y1),(0, 255, 0))
                if str(class_id) in Model["classes"]:
                    label = Model["classes"][str(class_id)] + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(y0, labelSize[1])
                    cv.rectangle(frame, (x0, y0 - labelSize[1]),(x0 + labelSize[0], y0 + baseLine),(255, 255, 255), cv.FILLED)
                    cv.putText(frame, label, (x0, y0),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        #cv.imshow("detections", frame)
        cv.imwrite("output/detections{}.png".format(uniqueStr()), frame)
        #if cv.waitKey(1) >= 0:
            #break

if __name__ == "__main__":
    main()
    
