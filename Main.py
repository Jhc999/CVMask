import cv2
import time
import argparse
import sys
import os
import numpy as np
import csv

from PIL import Image
from utils2 import *

from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

from imutils.video import FPS
import imutils

#################LOAD CSV&EMOJI##########################
with open('Results.csv', 'a') as textfile: 
    writer = csv.writer(textfile)
    writer.writerow(["ID","Y/N Mask", "Confidence"])

sizing = 30
imgHAPPY = cv2.imread("emoji_happy.png",-1)
imgMIDDLE = cv2.imread("emoji_middle.png",-1)
imgWORRIED = cv2.imread("emoji_worried.png",-1)
imgH = cv2.resize(imgHAPPY, (sizing, sizing))
imgM = cv2.resize(imgMIDDLE, (sizing, sizing))
imgW = cv2.resize(imgWORRIED, (sizing, sizing))

####################YOLOFACE CONFIG######################
'''
model_cfg = './cfg/yolov3-face.cfg'
model_weights = './model-weights/yolov3-wider_16000.weights'
netYolo = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
netYolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
netYolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
'''
#########################################################

######################YOLOv4 CONFIG######################
ap = argparse.ArgumentParser()

cfgfile = './cfg/yolov4-tiny.cfg'
weightfile = './yolov4-tiny.weights'
use_cuda = False  #GPU CONFIGm

m = Darknet(cfgfile)

#m.print_network()
m.load_weights(weightfile)
#print('Loading weights from %s... Done!' % (weightfile))

if use_cuda:
    m.cuda()
    
'''
num_classes = m.num_classes
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/x.names'
'''
namesfile = 'data/one.names'
class_names = load_class_names(namesfile)
   

# check if we are going to use GPU
maskGPU = False
if maskGPU == True:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#########################################################


####################MASKDETECT CONFIG####################
sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

#ENABLE GPU FOR MASK DETECTION
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# To run on GPU, set it to '0' or the GPU number
#########################################################

#####################HELPERS#############################
def allFalse(listy):
    for i in range (0, len(listy)):
        if listy[i] != False:
            return False
    return True

def intersect(a, b):
    #min right - max left
    #min bot   - max top
    dx = min(a[3],b[3]) - max(a[2],b[2])  
    dy = min(a[1],b[1]) - max(a[0],b[0])
    if (dx>=0) and (dy>=0):
        #print(dx*dy)
        return dx*dy
    else:
        return 0

def IU(a,b):
    if a == False or b == False:
        return float("inf")
    A = (a[0]-a[1])*(a[2]-a[3])
    B = (b[0]-b[1])*(b[2]-b[3])
    inter = intersect(a,b)
    return inter/(A+B-inter)

def minIU2(A,B):   #ACTUAL INTERSECT/UNION
    currmin = float("inf")
    Aind = -1
    Bind = -1
    for i in range (0, len(A)):
        for j in range (0, len(B)):
            if IU(A[i], B[j]) < currmin:
                currmin = IU(A[i], B[j])
                Aind = i
                Bind = j
    return (Aind, Bind, currmin)

def minIU(A,B):   #JUST TOP LEFT PIXEL
    #pixelList.append([top-oft, bottom+oft, left-oft, right+oft])
    currmin = float("inf")
    Aind = -1
    Bind = -1
    for i in range (0, len(A)):
        for j in range (0, len(B)):
            if A[i] != False and B[j] != False:
                topA = A[i][0]
                lefA = A[i][2]
                topB = B[j][0]
                lefB = B[j][2]
                dist = np.sqrt((topA-topB)**2+(lefA-lefB)**2)
                if dist < currmin:
                    currmin = dist
                    Aind = i
                    Bind = j
    return (Aind, Bind, currmin)

def ccw(A,B,C):
    #return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def boxXline(box, line):
#(startX, startY, endX, endY)
    topL  = [box[0],box[1]]
    topR  = [box[2],box[1]]
    botL  = [box[0],box[3]]
    botR  = [box[2],box[3]]
    line1 = [topL,topR]
    line2 = [topL,botL]
    line3 = [topR,botR]
    line4 = [botL,botR]
    lineL = [[line[0],line[1]],[line[2],line[3]]]
    if intersect(line1[0], line1[1], lineL[0], lineL[1]) == True:
        return True
    if intersect(line2[0], line2[1], lineL[0], lineL[1]) == True:
        return True
    if intersect(line3[0], line3[1], lineL[0], lineL[1]) == True:
        return True
    if intersect(line4[0], line4[1], lineL[0], lineL[1]) == True:
        return True
    return False

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

### Brightness and contrast

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

#########################################################

#################PARAMETER SETTINGS######################
brightSetting = 100      #amount of brightness increase
contrastSetting = 64     #amount of contrast increase
FLIR = False             #FLIR Camera, True = FlIR, False = Laptop
s = 128                  #size of display faces
display = [0,1,2,3,4]    #displayID of faces in separate window
nextEmpty = 0            #next displayID for side window
count = 100              #ID counter for tracking
addYolo = False          #add YoloFace for better face cropping
maxdistance = 200#1000        #the maximum two faces can be apart in two frames

#CROPPING THE PERSON BOUNDING BOX TO GET THE FACE

#CROPPING SETTINGS FOR OUTPUT VIDEO OF HENWEI AND 3 HOSPITAL STAFF
#topCUT = -0.1
#botCUT = -0.7#-0.1 for webcame in front of laptop
#leftCUT = +0.1
#rightCUT = -0.2

topCUT = 0.0
botCUT = -0.8
leftCUT = +0.1
rightCUT = -0.1

#HALLWAY TRACKING PARAMETERS
keepBOUNDx= 530#470#540    !!!!!!!!If startX < keepBOUND, the bounding box is added to pixelList, is candidate

WillWork = True
if WillWork == True:       #ACTUAL keepBOUND, bigger because give extra room for pairing (but these won't be kept for next iteration)
    keepBOUNDx = 900

keepBOUNDy= 2000#755       !!!!!!!!Not used
ridBOUNDx = 800#530        !!!!!!!!If startX < ridBOUNDx then it's a candidate for caching 
ridBOUNDy = 0#755        !!!!!!!!If startY > ridBOUNDy then it's a candidate for caching
numTimesAllowedInCache = 5 #!!!!!!!Max number of frames a candidate can be cached (allow up to 4 missing frames)
ridXoffset = 100#          !!!!!!!!If startX < ridBOUNDx-ridXoffset, it is ALWAYS cached to link in tracking

#ridBOUNDx = 2000

##########################################
#Iterating of none paired boxes from last frame
#if fLAST[i][0] < ridBOUNDx and fLAST[i][1] > ridBOUNDy:
##########################################


#lineline =(150,760,490,710)  #output2 or 3

#lineline = (0,600,800,600)
lineline = (0,350,1200,350)

debug = True           #True will show all boxes and lines on main video
highconfdetects = 1000    #number of highconfidence detections required      

####################LOAD VIDEO##########################
if FLIR == True:
    import EasyPySpin
    vidcap = EasyPySpin.VideoCapture(0)

if FLIR == False:
    #vidcap = cv2.VideoCapture(0)
    vidcap = cv2.VideoCapture("test_1.mp4")
#vidcap = cv2.VideoCapture(0)
vidcap.set(4,720)
has_frame,frame = vidcap.read()
fps = FPS().start()

out = np.zeros((s*5, s*1, 3), dtype = np.uint8)

fLAST = []
fCURR = []
idCURR = []
idLAST = []
mskLAST = []
mskCURR = []
IDdict = {}

numTimesCached={}

fpsCounter = 0

YesMask = 0
NoMask = 0

while has_frame:
    if FLIR == False:
        has_frame,frame = vidcap.read()  #webcam
        for i in range (0, 4):
            has_frame,frame = vidcap.read()
            
    if FLIR == True:
        has_frame,frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

################ IDENTIFY AND CROP FACE #################
    imageList = []
    pixelList = []
    faceList = []
    
    ############YOLOv4
    sized = cv2.resize(frame, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
    frame = imutils.resize(frame, width=1024)
    for i in range(0, len(boxes[0])):
        if boxes[0][i][6] == 0:
            boxBound = boxes[0][i]
            x1 = int(boxBound[0] * frame.shape[1])
            y1 = int(boxBound[1] * frame.shape[0])
            x2 = int(boxBound[2] * frame.shape[1])
            y2 = int(boxBound[3] * frame.shape[0])
            if debug == True:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255), 2)  #DRAW_PERSON_BOXES

            startX, endX, startY, endY = x1, x2, y1, y2

            tempWIDTH = endX - startX
            tempHEIGH = endY - startY
            top  = int(startY + topCUT*tempHEIGH)   #- 0.1*tempHEIGH) 
            bot  = int(endY   + botCUT*tempHEIGH)   #- 0.1*tempHEIGH)    #0.7
            left = int(startX + leftCUT*tempWIDTH)  #+ 0.0*tempWIDTH)  
            right= int(endX   - rightCUT*tempWIDTH) #- 0.2*tempWIDTH)
            crop_img = frame[top:bot, left:right]
            
            box = (startX, startY, endX, endY)
            if boxXline(box, lineline) and startX < keepBOUNDx: #and endY < keepBOUNDy:
            #if startY > 300 and startY <500 and startX < keepBOUNDx: #JUST FOR TESTING
                pixelList.append([startX,startY,endX,endY])
                #INCREASE IMAGE BRIGHTNESS
                #crop_img = apply_brightness_contrast(crop_img, 100, 64)
                crop_img = apply_brightness_contrast(crop_img, brightSetting, contrastSetting)
                imageList.append(crop_img)
                faceList.append([top,bot,left,right])
                if debug == True:
                    cv2.rectangle(frame, (left,top), (right,bot),255, 2)
                #cv2.rectangle(frame, (startX, startY), (endX, endY),255, 2)

    #cv2.line(frame, (keepBOUND, 760), (keepBOUND, 710), color=(0,255,0), thickness=4)
    #imageList, pixelList
    #pixelList = [top,bot,left,right]
    
### FACE TRACK
    if idLAST == []:
        for ll in range (0, len(pixelList)):
            idLAST.append(count)
            count+=1    #just here once confirmed
    if fLAST == []:
        fLAST = list.copy(pixelList)

    fCURR = []
    fCURR = list.copy(pixelList)
    backup = list.copy(pixelList)
    idCURR = []
    for a in range (0, len(fCURR)):
        idCURR.append(-1)

    for b in range (0, len(fCURR)):              
        #NO, CANDIDATES LEFT, CREATE NEW IDS
        #print(fCURR, fLAST, allFalse(fCURR), allFalse(fLAST))
        if (allFalse(fCURR) == False) and (allFalse(fLAST) == True):
            #CREATE NEW IDS
            for c in range (0, len(fCURR)):
                if fCURR[c] != False:
                    idCURR[c] = count
                    #print("flag1")
                    count += 1     #INCREASE

        #if allfalse(fCURR) == True, it means everything has been paired with an ID
        #don't need to continue linking
        
        #CANDIDATES LEFT
        #print("ONE")
        if (allFalse(fCURR) == False) and (allFalse(fLAST) == False):
            (lastIND, currIND, currmin) = minIU(fLAST, fCURR)
            #print("HALF")
            #print(fLAST, fCURR, idLAST, idCURR)
            fLAST[lastIND] = False
            #print(fLAST, fCURR, idLAST, idCURR)
            fCURR[currIND] = False
            #print(fLAST, fCURR, idLAST, idCURR)

            #print(currmin)
            #print(currmin)
            if currmin < maxdistance:
                #print(idCURR, idLAST, currIND, lastIND)
                idCURR[currIND] = idLAST[lastIND]
            if currmin >= maxdistance: 
                idCURR[currIND] = count
                #print("flag2")
                count += 1   #INCREASE
            #print(fLAST, fCURR, idLAST, idCURR)
        #print(fCURR)
        #print(idCURR)                   
    
    #######START Current tracking fix: if flast has an unmatched one, don't delete
    for i in range (0, len(idCURR)):
        #cv2.rectangle(frame, (backup[i][0], backup[i][1]),
        #              (backup[i][2], backup[i][2]),255, 2)
        if debug == True:
            cv2.putText(frame, str(idCURR[i]), (backup[i][0],backup[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (244,0,255), 5)

    
    for i in range (0, len(fLAST)):
        if fLAST[i] != False:  #there's an unmatched frame

    #pixelList.append([startX,startY,endX,endY])
    #flast gets pixellist
            if fLAST[i][0] < ridBOUNDx and fLAST[i][1] > ridBOUNDy:  #the unmatched frame didn't exit
                #print(fLAST[i][0], fLAST[i][1], ridBOUNDx)

                ###HAVE CODE WHERE IF fLAST[i][0] < ridBOUNDx-100 THEN ALWAYS CACHE
                ###ADD ALSO A HORIZONTAL LINE (For people going other way in the hallway)
                ###1. CLEAN CODE TOP TO BOT (10min)
                ###2. GATHER ALL PARAMETERS (10min)
                ###3. SIDEBAR, LIST AND EMOJIS \
                ###4. ADD IN YOLOFACE

                if fLAST[i][0] < ridBOUNDx - ridXoffset:
                    idCURR.append(idLAST[i])
                    backup.append(fLAST[i])                
                elif idLAST[i] not in numTimesCached:
                    idCURR.append(idLAST[i])
                    backup.append(fLAST[i])
                    numTimesCached[idLAST[i]] = 1
                else:
                    if numTimesCached[idLAST[i]] < numTimesAllowedInCache:    
                        idCURR.append(idLAST[i])
                        backup.append(fLAST[i])
                        numTimesCached[idLAST[i]] += 1

    #if debug == True:           
    #    cv2.line(frame, (ridBOUNDx, 760), (ridBOUNDx, 100), color=(0,0,255), thickness=4)
    #cv2.line(frame, (0, keepBOUNDy), (800, keepBOUNDy), color=(0,0,0), thickness=4)
    #print(idCURR)

    #######END Current tracking fix
    
    idLAST = list.copy(idCURR)
    fLAST  = list.copy(backup)

    if WillWork == True:   #THIS IS THE TRACKING FIX!!!! KEEP IT FOR PAIRING, BUT REMOVE IT FROM CANDIDATES NEXT ROUND
        for t in range (len(idLAST), -1):
            if fLAST[t][0] >= ridBOUNDx:
                fLAST.pop(t)
                idLAST.pop(t)

    if debug == True:
        cv2.line(frame, (lineline[0],lineline[1]), (lineline[2],lineline[3]), color=(255,255,255), thickness=4)

#####################SELECTING FACES####################
    for i in range (0, len(idCURR)):
        #IDdict[100] = [count, YES/NO/NONE, confidence, highconfCOUNT, Yes/No mask counter]
        candidates = []
        if idCURR[i] in IDdict:
            if 1 == 1:
            #if IDdict[idCURR[i]][1] != "YES":
                if IDdict[idCURR[i]][0] < highconfdetects:     #NUMBER OF HIGH CONF DETECTIONS REQUIRED
                    #print("STOP PREDICTION")
                    candidates.append(idCURR[i])
                

        else:
            candidates.append(idCURR[i])
            IDdict[idCURR[i]] = [0, "NONE", 0, 0, False]
    #if 102 in IDdict:
        #print(102, 102, IDdict[102])

    #Mask detect if ID is in candidates
    #Process mask detection at the end
    #Display
    #Test mask detection, then decide if we need 1) better detector 2) yolo in the middle

    thresholdCONF = 0.70
########################################################


########################  MASK DETECT ##################
    #Call inference() on each of the faces in imageList
    doneImage = []
    #wearMask
    for i in range (0, len(imageList)):
        if idCURR[i] in candidates:
        #cv2.imshow("", imageList[i])
            try:
                
                
                ##################ADDITIONAL FACE DETECTION YOLO########
                '''
                if addYolo == True:
                    blobby = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                         [0, 0, 0], 1, crop=False)
                    netYolo.setInput(blobby)
                    outs = netYolo.forward(get_outputs_names(netYolo))
                    faces, outImage, outPixel = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
                    #if outImage != []:
                        #imageList[i] = outImage[0]
                '''

                image = imageList[i]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #
                #(hEIGHTT, widthHH) = image.shape[:2]
                #image = imutils.resize(image, width=600)
                #image = imutils.resize(image, width=widthHH)
                #
                conf_thresh = 0.5
                iou_thresh = 0.4
                #(h, w) = frame.shape[:2]
                target_shape = (260, 260)
                draw_result = True
                show_result = True

                output_info = []
                height, width, _ = image.shape
                image_resized = cv2.resize(image, target_shape)
                #cv2.imshow("", image)
                image_np = image_resized / 255.0  
                image_exp = np.expand_dims(image_np, axis=0)
                y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

                # remove the batch dimension, for batch is always 1 for inference.
                y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
                y_cls = y_cls_output[0]
                # To speed up, do single class NMS, not multiple classes NMS.
                bbox_max_scores = np.max(y_cls, axis=1)
                bbox_max_score_classes = np.argmax(y_cls, axis=1)

                # keep_idx is the alive bounding box after nms.
                keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                             bbox_max_scores,
                                                             conf_thresh=conf_thresh,
                                                             iou_thresh=iou_thresh,
                                                             )
               
                
                for idx in keep_idxs:
                    conf = float(bbox_max_scores[idx])
                    class_id = bbox_max_score_classes[idx]
                    bbox = y_bboxes[idx]
                    # clip the coordinate, avoid the value exceed the image boundary.
                    xmin = max(0, int(bbox[0] * width))
                    ymin = max(0, int(bbox[1] * height))
                    xmax = min(int(bbox[2] * width), width)
                    ymax = min(int(bbox[3] * height), height)
                    '''
                    if 100 in IDdict:
                        print(100, IDdict[100])
                    if 101 in IDdict:
                        print(101, IDdict[101])
                    if 103 in IDdict:
                        print(103, IDdict[103])
                    '''
                    print(idCURR[i], conf)
                    if draw_result:
                        if class_id == 0 and conf >=thresholdCONF:
                            color = (0, 255, 0)
                            img = imgH  #happy
                            valMASK = "YES"
                            IDdict[idCURR[i]][2] += conf
                            IDdict[idCURR[i]][3] += 1
                        if class_id == 1 and conf >=thresholdCONF:
                            color = (255, 0, 0)
                            img = imgW  #worried
                            valMASK = "NO"
                            IDdict[idCURR[i]][2] -= conf
                            IDdict[idCURR[i]][3] += 1
                        if conf < thresholdCONF: #0.70
                            color = (255, 255, 0)
                            img = imgM  #middle
                            valMASK = "MAYBE"
                IDdict[idCURR[i]][0] += 1
                IDdict[idCURR[i]][1] = valMASK
                valMASK = "NONE"
                #if idCURR[i]==101:                
                    #cv2.namedWindow("Channel1")
                    #cv2.imshow("Channel1",imageList[i])
                    #cv2.waitKey(1)
                
                '''
                        #FACE DRAWING
                        
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 10),  #ymin+10
                                    #cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
                        cv2.putText(image, "ID: %s" % (idCURR[i]), (xmin + 2, ymax + 10),  #ymax-4
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
                '''
                        ############

                '''                       
                        #x_offset=y_offset=75
                        #y1, y2 = y_offset, y_offset + img.shape[0]
                        #x1, x2 = x_offset, x_offset + img.shape[1]

    #                    try:
    #                        y1, y2 = ymax-img.shape[0], ymax
    #                        x1, x2 = xmax-img.shape[0], xmax
    #                        alpha_s = img[:, :, 3] / 255.0
    #                        alpha_l = 1.0 - alpha_s
    #                        for c in range(0, 3):
    #                            image[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
    #                    except:
    #                        print("Emoji Error")

                        
                        #with open('Results.csv', 'a') as textfile: 
                            #writer = csv.writer(textfile)
                            #writer.writerow([idCURR[i],id2class[class_id],conf])
                    output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
                '''
                #cv2.imshow("show", image)
                #cv2.waitKey(1)
                #if show_result:
                    #Image.fromarray(image).show()
                    #Image2 = np.array(Image)
                    #cv2.imwrite('one.jpg', Image2)  
                    #cv2.imshow("show", Image2)
                    #cv2.waitkey(1)
                #EMOJI
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                '''
                try:
                    y1, y2 = ymax-img.shape[0], ymax
                    x1, x2 = xmax-img.shape[0], xmax
                    alpha_s = img[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(0, 3):
                        image[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
                except:
                    print("Emoji Error")
                top  = faceList[i][0]
                bot  = faceList[i][1]
                left = faceList[i][2]
                right= faceList[i][3]
                #image = imutils.resize(image, width=widthHH, height=hEIGHTT)
                print("GETTING TO THIS POINT")
                frame[top:bot, left:right] = image
                '''
            except:
                print("Exception")
                continue
    #cv2.imshow("show", frame)
    #cv2.waitKey(1)


    # initialize the set of information we'll displaying on the frame
#    info = [('number of faces detected', '{}'.format(len(faces)))]
#    for (i, (txt, val)) in enumerate(info):
#        text = '{}: {}'.format(txt, val)
#        cv2.putText(frame, text, (10, (i * 20) + 20),
#            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

########DRAWING#################################
    #BGR for cv2
    for i in range (0, len(idCURR)):
        finalCONF = 0.00
        if IDdict[idCURR[i]][3] != 0:
            finalCONF = (IDdict[idCURR[i]][2]/IDdict[idCURR[i]][3])*10           
            finalCONF = int((finalCONF * 100) + 0.5) / 100.0

        '''    
        #ASSUMING NO FALSE POSITIVE
        if IDdict[idCURR[i]][1] == "YES":     #GREEN
            DRAWcolor = (0, 255, 0)
        if IDdict[idCURR[i]][1] == "NO":      #RED
            DRAWcolor = (0, 0, 255)      
        if IDdict[idCURR[i]][1] == "NONE":    #BLUE
            DRAWcolor = (255, 0, 0)
        if IDdict[idCURR[i]][1] == "MAYBE":   #YELLOW
            DRAWcolor = (255, 255, 0)
        '''

        #BY SCORE
        if finalCONF > 0:     #GREEN
            DRAWcolor = (0, 255, 0)
        if finalCONF < 0:      #RED
            DRAWcolor = (0, 0, 255)      
        if finalCONF == 0.00:    #YELLOW
            DRAWcolor = (0, 255, 255)
        #else:  #BLUE
        #    DRAWcolor = (255, 255, 0)
            
        try:
            [Xstart, Ystart, Xend, Yend] = pixelList[i]
            if debug == True:
                cv2.rectangle(frame, (Xstart, Ystart), (Xend, Yend), DRAWcolor, 2)

            countnumber = str(IDdict[idCURR[i]][0]) 
            if IDdict[idCURR[i]][0] == 5:
                countnumber = "Done"
            
            #cv2.putText(frame, countnumber, (pixelList[i][0]+1, pixelList[i][1]+15),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, DRAWcolor, 2)
            if debug == True:
                cv2.putText(frame, "Score: "+str(finalCONF), (pixelList[i][0]+1, pixelList[i][1]+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAWcolor, 2)
            
        except:
            continue

    ###CALCULATE COUNTER
    '''
    try:
        #temp = []
        for testID in IDdict:
            #temp.append(testID)
            
            if testID not in idCURR:
                finalCONF = (IDdict[testID][2]/IDdict[testID][3])*10           
                finalCONF = int((finalCONF * 100) + 0.5) / 100.0
                if finalCONF > 0 and IDdict[testID][4] == False and IDdict[testID][0] == 5:     #GREEN
                    YesMask+=1
                    IDdict[testID][4] = True
                if finalCONF < 0 and IDdict[testID][4] == False and IDdict[testID][0] == 5:      #RED
                    NoMask+=1
                    IDdict[testID][4] = True
        #print(temp, idCURR)
        cv2.putText(frame, "Yes Masks: "+str(YesMask), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) , 3)
        cv2.putText(frame, "No  Masks: "+str(NoMask), (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) , 3)
    except:
        print("Counter Error")
        continue
    '''

    ##### DELETE DOWN
    for testID in IDdict:
            
        if testID not in idCURR:
            finalCONF = 0
            if IDdict[testID][3] != 0:
                finalCONF = (IDdict[testID][2]/IDdict[testID][3])*10           
                finalCONF = int((finalCONF * 100) + 0.5) / 100.0
            if finalCONF > 0 and IDdict[testID][4] == False and IDdict[testID][0] == highconfdetects:     #GREEN
                YesMask+=1
                IDdict[testID][4] = True
            if finalCONF < 0 and IDdict[testID][4] == False and IDdict[testID][0] == highconfdetects:      #RED
                NoMask+=1
                IDdict[testID][4] = True
        #print(temp, idCURR)
    cv2.putText(frame, "Yes Masks: "+str(YesMask), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) , 3)
    cv2.putText(frame, "No  Masks: "+str(NoMask), (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) , 3)

    
    ##### DELETE UP
    
    ####
    cv2.namedWindow("Main")
    cv2.imshow("Main", frame)
    cv2.waitKey(1)
    
    try:
        cv2.namedWindow("Channel2")
        for i in range (0, min(5, len(imageList))):
            #row = s*int(i/3)
            #col = s*(i%3)
            if idCURR[i] not in display:
                ii = nextEmpty % 5
                display[ii] = idCURR[i]
                nextEmpty+=1
            else:
                ii = display.index(idCURR[i])
            row = s*int(ii/1)
            col = s*(ii%1)

            ####ISSUE WITH RESIZING EMPTY IMAGES
            faceimg = cv2.resize(imageList[i], (s,s), 0, 0, cv2.INTER_AREA)
            finalCONF = 0.00
            if IDdict[idCURR[i]][3] != 0:
                finalCONF = (IDdict[idCURR[i]][2]/IDdict[idCURR[i]][3])*10           
                finalCONF = int((finalCONF * 100) + 0.5) / 100.0
            if finalCONF > 0:     #GREEN
                DRAWcolor = (0, 255, 0)
                emoji = imgH
            if finalCONF < 0:      #RED
                DRAWcolor = (0, 0, 255)
                emoji = imgW
            if finalCONF == 0.00:    #YELLOW
                DRAWcolor = (0, 255, 255)
                emoji = imgM

            #ADD EMJOI
            xmax = s-5
            ymax = s-5
            try:
                y1, y2 = ymax-emoji.shape[0], ymax
                x1, x2 = xmax-emoji.shape[0], xmax
                alpha_s = emoji[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    faceimg[y1:y2, x1:x2, c] = (alpha_s * emoji[:, :, c] + alpha_l * faceimg[y1:y2, x1:x2, c])
            except:
                print("Emoji Error")
            #DONE EMOJI
        
            cv2.rectangle(faceimg, (1,1), (127,127), DRAWcolor, 2)
            if debug == True:
                cv2.putText(faceimg, str(idCURR[i]), (1,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            out[row:row+s, col:col+s] = faceimg
        cv2.imshow("Channel2", out)
    except:
        print("Continue Side Display")
        continue
    
################################################
    fps.update()
    fpsCounter += 1
    if fpsCounter == 15:
        fpsCounter = 0
    fps.stop()
    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    fps = FPS().start()
cv2.destroyAllWindows()


