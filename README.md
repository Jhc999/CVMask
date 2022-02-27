# The Acceptance of a Computer Vision Facilitated Protocol to Measure Adherence to Face Masks

#### Journal Paper in ???, Volume ???, ???: LINK ####

This repository provides the code and data used in the paper.

## Setup
1. Nvidia Jetson Xavier NX
2. FLIR Chameleon Camera
3. FLIR Spinnaker SDK
4. YoloV4-tiny model for person tracking
5. YoloV4 model for mask detection *(best)
6. Tensorflow model for mask detection *(outdated) 

*Results from the paper are obtained with Nvidia Jetson, FLIR Chameleon Camera, and YoloV4 model for mask detection. 
*For convenience, code can be tested on a PC with a compatible camera/webcam.

## Dependencies
1. Python-3.6
2  Opencv-3.4.6
3. Imutils-0.5.3
4. Pillow-8.0.2
5. Numpy-1.16.1
6. EasyPySpin-1.1.0
8. Pytorch-1.6.0
9. Darknet-0.3 
10. Tensorflow-2.3.0 (if using Tensorflow model for mask detection)

## Person Tracker / Mask Detector 

* ```Main.py```             Main code to deploy person tracker / mask detector
* ```./tool```              Darknet helpers
* ```./utils```             YoloV4 helpers
* ```utils2.py```           Mask detector configuration file
* ```Imgs```                Image icons for feedback

## Training Mask Detector

* YoloV4 detection model is trained for mask detection, training code available upon request

## Trained Models 
* ```./cfg```               Yolov4-tiny config files
* ```yolo4-tiny.weights```  Yolov4-tiny weights
* ```./models```            Tensorflow models for mask detection  
* YoloV4 mask detection model available upon request
