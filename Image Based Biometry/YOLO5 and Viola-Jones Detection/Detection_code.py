# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 17:44:58 2022

@author: Asus
"""

import glob
import os
import copy
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt  
    
#%% FUNCTIONS

"""
Function read_img_txt() : 
    - reads all 500 images from the database and saves them in a list of 500 elements
    - reads all txt files with ground truths and saves them in a list of 500 elements
    - converts ground truths values from: x-center, y-center width and height in normalized
      form to x and y coordinates of top left pixel, width and height in pixel values
"""

def read_img_txt():
    
    images = []
    gray_images = []
    ground_truth = []
    gt_box = []
    path ="C:/Users/Asus/Desktop/biometry/assignment2/SupportFiles/ear_data/test"
    img_path = os.path.join(path,'*png')
    txt_path = os.path.join(path,'*txt')
        
    for file in glob.glob(img_path):
        input_img = cv2.imread(file)                                            # reads an image 
        images.append(input_img)                                                # appends image to a list
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)  
        gray_images.append(gray_img)
        
    for file in glob.glob(txt_path):
        f = open(file, 'r')                                                     # opens txt file with ground truths
        content = f.read()                                                      # reads txt file
        f.close()
        content = content.split()                                               # separates values and creates a list
        content = content[1:]                                                   # leaves out the first value (0 in all the files)
        ground_truth.append(content)
    
    for i in range(0, len(ground_truth)):
        gt_box_i = [0, 0, 0, 0]
        img_height, img_width = images[i].shape[:2]                             # returns image size in pixels
        gt_box_i[2] = round(float(ground_truth[i][2])*img_width)                # converts normalized width to width in pixels
        gt_box_i[3] = round(float(ground_truth[i][3])*img_height)               # converts normalized height to height in pixels
        gt_box_i[0] = round(float(ground_truth[i][0])*img_width-gt_box_i[2]/2)  # calculates x coordinate of top-left pixel
        gt_box_i[1] = round(float(ground_truth[i][1])*img_height-gt_box_i[3]/2) # calculates y coordinate of top-left pixel
        gt_box.append(gt_box_i) 
    
    return gray_images, images, gt_box  


"""
Function ear_detection(gray_images, images, sF, mN, mS):
    - takes arguments: list of greyscale images, list of color images, 
      scale factor, minimum neighbors and minimum possible object size for VJ detection
    - implements VJ algorithm for ear detection on grayscale images and draws detection boxes on color images
    - returns list of 500 elements where every element is a smaller list that contains 
      x,y coordinates of top-left pixel, width and height of detection box for 1 image 
    - returns list of images with detection boxes
"""

def ear_detection(gray_images, images, sF, mN, mS):
    
    VJ_box = []
    
    for i in range (0, len(images)):
        
        img_box_l = []
        img_box_r = []
        img_box = []
        
        # runs VJ algorithm with chosen parameters
        left_ear = left_ear_cascade.detectMultiScale(gray_images[i], scaleFactor = sF, minNeighbors = mN, minSize = mS)
        right_ear = right_ear_cascade.detectMultiScale(gray_images[i], scaleFactor = sF , minNeighbors = mN, minSize = mS)
        
        # makes a list of detection box values for all detections in an image: x,y coordinate sof top-left pixel, width and height
        if len(left_ear) != 0:                                                  
            for j in range(0,len(left_ear)):                                    
                img_box_l = [x for x in left_ear[j].tolist()]
                img_box.append(img_box_l)                                       
        
        if len(right_ear) != 0:
            for j in range(0,len(right_ear)):
                img_box_r = [x for x in right_ear[j].tolist()]
                img_box.append(img_box_r)
        
        # appends detection box values of one image to the list of detection box values for all images 
        VJ_box.append(img_box)
        
        # sets different rectangle line thickness for different image size
        if images[i].size > 10000000:                                           
            line_tick = 16                                                      
        elif images[i].size > 1000000:
            line_tick = 8
        elif images[i].size > 100000:
            line_tick = 4
        else: line_tick = 2
        
        # draws detection boxes (rectangles) 
        for (x, y, w, h) in left_ear:                                           
            cv2.rectangle(images[i], (x, y), (x + w, y + h), (0, 0, 255), line_tick)
        for (x, y, w, h) in right_ear:
            cv2.rectangle(images[i], (x, y), (x + w, y + h), (0, 0, 255), line_tick)
    
    return VJ_box, images

"""
Function intersection_over_union(boxA, boxB):
    - takes as input arguments 2 boxes: ground-truth box (A) and VJ detection box (B)
    - calculates Intersection over Union for every detection in an image
    - returns a list of 500 elements where every element is a list of IoU value/s for detection/s in one image
"""

def intersection_over_union(boxA, boxB):
    
    iou = []
    
    if len(boxB) == 0:
        iou_i = 'no detection'
        iou.append(iou_i)
    else:
        for i in range(0, len(boxB)):
            Ax1 = boxA[0]
            Ay1 = boxA[1]
            Aw = boxA[2]
            Ah = boxA[3]
            # finding x and y coordinates of bottom right pixel in ground truth box
            Ax2 = Ax1 + Aw
            Ay2 = Ay1 + Ah
            
            Bx1 = boxB[i][0]
            By1 = boxB[i][1]
            Bw = boxB[i][2]
            Bh = boxB[i][3]
            # finding x and y coordinates of bottom right pixel in detection box
            Bx2 = Bx1 + Bw
            By2 = By1 + Bh
            
            # calculates lenghts of overlapping rectangle sides
            delta_x = max(Ax1, Bx1) - min(Ax2, Bx2)
            delta_y = max(Ay1, By1) - min(Ay2, By2)
            
            # calculates area of intersection
            overlap = delta_x * delta_y
            
            # calculates union of 
            try:
                union = Aw * Ah + Bw * Bh - overlap
                iou_i = overlap / union
                iou.append(iou_i)
            except ZeroDivisionError:
                iou.append('zero division')
            
    return iou

"""
Function accuracy (threshold, iou):
    - takes arguments: threshold value(if obtained IoU value is higher than threshold it is considered true positive)
      and list of Intersection-over-Union values for all detections
    - makes a list of  all FNs, TPs, FPs and calculates accuracy
    - returns number of FNs, TPs, FPs and accuracy
"""
def accuracy(threshold, iou):
    
    TP = []
    FP = []
    FN = []
    
    for value in iou :
        if value [0] == 'no detection':                                         # if there is no detection in an image it is considered false negative
            FN.append(value)                                                    # appends 'no detection' to FN list 
        else:
            for i in range(0,len(value)):                                       # for every detection in an image  
                if value[i] >= threshold and value[i] < 1:                      # checks if IoU value is higher than a threshold
                    TP.append(value)                                            # if value is higher or equal appends it to TP list
                else:
                    FP.append(value)                                            # if value is lower appends it to FP list 
                    
    print(len(FN),len(TP), len(FP))
               
    ACC = len(TP)/(len(TP)+len(FP)+len(FN))                                     # calculates accuracy
    
    return len(FN), len(TP), len(FP), ACC   

"""
Function average_iou:
    - takes as input argument list of IoU values for all detections
    - calculates and returns average IoU for all detections
"""

def average_iou (iou):
    total = 0
    num = 0
            
    for v in iou:
        if v[0] == 'no detection':
            num += 1
            continue
        else:
            for i in range(0, len(v)):
                if v[i] > 1 or v[i] < 0:
                    num += 1
                    continue
                else:
                    total = total + v[i]
                    num += 1
    average_iou = total / num  
    
    return average_iou

"""
Function VJ_detection(gray_images, images, gt_box, sF, mN, mS):
    - takes following input arguments: list of grayscale images, list of images in color, 
      list of ground-truth boxes for every image, scale factor,  minimum neighbors, 
      minimum possible object size for VJ detection and threshold value
    - calls ear_detection function that implements VJ algorithm on input images
    - calls intersection_over_union function that calculates IoU for every detection
    - calls accuracy function that calculates accuracy of detection
    - returns accuracy of detection for specified threshold, list of IoU values for each detection and 
      list of images with detection rectangles
    
"""
def VJ_detection(gray_images, images, gt_box, sF, mN, mS, threshold):

    images_c = copy.deepcopy(images)
    
    # calling function that implements VJ algorithm on input images
    VJ_box, VJ_images = ear_detection(gray_images, images_c, sF, mN, mS)
    
    # calling IoU calculation for detections in every image
    iou = []
    for i in range (0,len(VJ_box)):
        iou_i = intersection_over_union(gt_box[i], VJ_box[i])
        iou.append(iou_i)

    # calculating average IoU value for all detections
    average_IoU = average_iou(iou)
    
    # calling accuracy calculation
    FN, TP, FP, ACC = accuracy(threshold, iou)
    
    print("parameters:", sF, mN, mS, 'accurancy:', ACC, 'Average IoU:', average_IoU)
    
    return ACC, iou, VJ_images

"""
Function precision_recall(iou):
    - takes as an input argument list of IoU values for all images
    - creates a list of 100 thresholds (from 0 to 1 with step 0.01)
    - for every threshold call accuracy function which returns number of TPs, FPs and FNs
    - calculates precision and recall for every threshold 
    - returns list of precision and list of recall values for all thresholds
"""

def precision_recall (iou):
    
    precision = []
    recall = []
    FP_num = []
    TP_num = []
    
    threshold = [] 
    j = 0
    for i in range(0,101):
        threshold.append(j)
        j += 0.01
            
    for t in threshold:
        FN, TP, FP, ACC = accuracy(t, iou)
        TPR = TP/(TP+FN)
        PR = TP/(TP+FP)
        FP_num.append(FP)
        TP_num.append(TP)
        precision.append(PR)
        recall.append(TPR)
   
    return precision, recall

"""
Function load_YOLO():
    - loads labels from YOLO detection method and convertes them to suitable form
    - inserts empty lists for images with no detection
    - for every detection calls function that calculates IoU
    - calls function that calculates average IoU
    - calls function that calculates accuracy
    - returns accuracy, average IoU and list of IoU values
    
"""
def load_YOLO():
    
    yolo_labels = []
    yolo_box = []
    
    location ="C:/Users/Asus/Desktop/biometry/assignment2/yolov5/runs/detect/exp8/labels"
    txt_location = os.path.join(location,'*txt')
           
    for file in glob.glob(txt_location):
           f = open(file, 'r')                                                     # opens txt file 
           content = f.read()                                                      # reads txt file
           f.close()
           content = content.split()                                               # separates values and creates a list
           content = content[1:]
           yolo_labels.append(content)
           
    
    for i in range(0, len(yolo_labels)):
        yolo_box_i = [0, 0, 0, 0]
        img_height, img_width = images[i].shape[:2]                                # returns image size in pixels
        yolo_box_i[2] = round(float(yolo_labels[i][2])*img_width)                  # converts normalized width to width in pixels
        yolo_box_i[3] = round(float(yolo_labels[i][3])*img_height)                 # converts normalized height to height in pixels
        yolo_box_i[0] = round(float(yolo_labels[i][0])*img_width-yolo_box_i[2]/2)  # calculates x coordinate of top-left pixel
        yolo_box_i[1] = round(float(yolo_labels[i][1])*img_height-yolo_box_i[3]/2) # calculates y coordinate of top-left pixel
        yolo_box.append([yolo_box_i])
    
    # calculates IoU for yolo detection boxes
    iou_yolo= []
    for j in range (0,len(yolo_box)):
        iou_i = intersection_over_union(gt_box[j], yolo_box[j])
        iou_yolo.append(iou_i)
    
    iou_yolo[361] = ['no detection']
    iou_yolo[369] = ['no detection']
    iou_yolo[378] = ['no detection']
    iou_yolo[384] = ['no detection']
    iou_yolo[392] = ['no detection']
    
    # calculating average IoU value for all detections
    average_IoU = average_iou(iou_yolo)
    
    # calling accuracy calculation
    FN, TP, FP, ACC = accuracy(0.5, iou_yolo)
    
    print('accurancy:', ACC, 'Average IoU:', average_IoU)
    
    return ACC, average_iou, iou_yolo

"""
Function best_VJ_predictions(iou, VJ_images, num):
    - takes following input arguments: list of IoU values, list of output images 
      of VJ detection with rectangles and wanted number of best predictions (num)
    - finds highest IoU value in the list, plots the images to which highest IoU value
      belongs and saves that IoU value to new list max_iou and removes it from IoU list
    - repeats previous procedure for specified number of times (num)
    - returns a list of maximum IoU values with num elements
    
"""
def best_VJ_predictions(iou, VJ_images, num):
    
    max_iou = []
    
    for n in range(0, num):
        max_v = 0
        for v in iou:
            if v[0] == 'no detection' or len(v) == 0:
                continue
            else:
                for i in range(0, len(v)):
                    if v[i] > max_v and v[i] < 1:
                        max_v = v[i]
                        max_img = iou.index(v)
        
        max_iou.append(max_v)
        iou.pop(max_img)
        plt.figure()
        plt.imshow(cv2.cvtColor(VJ_images[max_img], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        VJ_images.pop(max_img)
        print(max_img,max_v)
        
    return max_iou
"""
Function failed_prediction:
    - takes following input argumnets: list of IoU values, VJ output images 
    - plots images with no detection 
"""
def failed_predictions(iou, VJ_images):

    for v in iou:   
        if (v[0] == 'no detection') or (len(v) == 0):
            max_img = iou.index(v)
    
            iou.pop(max_img)
            plt.figure()
            plt.imshow(cv2.cvtColor(VJ_images[max_img], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            VJ_images.pop(max_img)
    
       
    
#%% MAIN CODE

# loading the images and txt files
gray_images, images, gt_box = read_img_txt()

# loading XML files
left_ear_cascade = cv2.CascadeClassifier('C:/Users/Asus/Desktop/biometry/assignment2/SupportFiles/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('C:/Users/Asus/Desktop/biometry/assignment2/SupportFiles/haarcascade_mcs_rightear.xml')

# VJ detection
VC_acc, VJ_IoU, VJ_images = VJ_detection (gray_images, images, gt_box, 1.025, 5, (30,30), 0.5)

VC_acc_3, VJ_IoU_3, VJ_images_3 = VJ_detection (gray_images, images, gt_box, 1.025, 3, (0,0), 0.5)
VC_acc_4, VJ_IoU_4, VJ_images_4 = VJ_detection (gray_images, images, gt_box, 1.025, 4, (0,0), 0.5)
VC_acc_5, VJ_IoU_5, VJ_images_5 = VJ_detection (gray_images, images, gt_box, 1.025, 5, (0,0), 0.5)

VC_acc_11, VJ_IoU_11, VJ_images_11 = VJ_detection (gray_images, images, gt_box, 1.1, 5, (0,0), 0.5)
VC_acc_1025, VJ_IoU_1025, VJ_images_1025 = VJ_detection (gray_images, images, gt_box, 1.025, 5, (0,0), 0.5)
VC_acc_175, VJ_IoU_175, VJ_images_175 = VJ_detection (gray_images, images, gt_box, 1.75, 5, (0,0), 0.5)

VC_acc_0, VJ_IoU_0, VJ_images_0 = VJ_detection (gray_images, images, gt_box, 1.025, 5, (0,0), 0.5)
VC_acc_15, VJ_IoU_15, VJ_images_15 = VJ_detection (gray_images, images, gt_box, 1.025, 5, (15,15), 0.5)
VC_acc_30, VJ_IoU_30, VJ_images_30= VJ_detection (gray_images, images, gt_box, 1.025, 5, (30,30), 0.5)
VC_acc_45, VJ_IoU_45, VJ_images_45= VJ_detection (gray_images, images, gt_box, 1.025, 5, (45,45), 0.5)

# Precision-recall
yolo_PR, yolo_R = precision_recall (yolo_iou)
precision, recall = precision_recall (VJ_IoU)

PR_3, R_3 = precision_recall(VJ_IoU_3)
PR_4, R_4 = precision_recall(VJ_IoU_4)
PR_5, R_5 = precision_recall(VJ_IoU_5)

PR_11, R_11 = precision_recall(VJ_IoU_11)
PR_1025, R_1025 = precision_recall(VJ_IoU_1025)
PR_175, R_175 = precision_recall(VJ_IoU_175)

PR_0, R_0 = precision_recall(VJ_IoU_0)
PR_15, R_15 = precision_recall(VJ_IoU_15)
PR_30, R_30 = precision_recall(VJ_IoU_30)
PR_45, R_45 = precision_recall(VJ_IoU_45)

# best VJ predictions
max_iou = best_VJ_predictions(VJ_IoU, VJ_images, 20)

# failed VJ predictions
failed_predictions(VJ_IoU, VJ_images)

failed_predictions(VJ_IoU_30, VJ_images_30)

#%% YOLO

!python C:/Users/Asus/Desktop/biometry/assignment2/yolov5/detect.py --weights C:/Users/Asus/Desktop/biometry/assignment2/SupportFiles/yolo5s.pt --img 416 --save-txt --save-conf --conf 0.4 --source C:/Users/Asus/Desktop/biometry/assignment2/SupportFiles/ear_data/test

ACC_yolo, average_IoU_yolo, yolo_iou = load_YOLO()

#%% VJ PARAMETAR OPTIMIZATION

for i in (0, 15, 30):
    for j in (3, 4, 5, 6):
        for k in (1.025, 1.05, 1.1):
            par_o = VJ_detection (gray_images, images, gt_box, k, j, (i,i), 0.5)

#%% Plotting

plt.figure()
plt.plot(yolo_R, yolo_PR, '--r')
plt.plot(PR_3, R_3, 'g')
plt.plot(PR_4, R_4, 'black')
plt.plot(PR_5, R_5, 'b')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(['YOLO','VJ: sF = 1.025, mN = 3, mS = (0,0)', 'VJ: sF = 1.025, mN = 4, mS = (0,0)', 'VJ: sF = 1.025, mN = 5, mS = (0,0)'])

plt.figure()
plt.plot(yolo_R, yolo_PR, '--r')
plt.plot(PR_175, R_175, 'g')
plt.plot(PR_11, R_11, 'b')
plt.plot(PR_1025, R_1025, 'black')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(['YOLO','VJ: sF = 1.75, mN = 5, mS = (0,0)', 'VJ: sF = 1.10, mN = 5, mS = (0,0)', 'VJ: sF = 1.025, mN = 5, mS = (0,0)'])

plt.figure()
plt.plot(yolo_R, yolo_PR, '--r')
plt.plot(PR_0, R_0, 'g')
plt.plot(PR_15, R_15, 'orange')
plt.plot(PR_30, R_30, 'black')
plt.plot(PR_45, R_45, 'b')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(['YOLO','VJ: sF = 1.025, mN = 5, mS = (0,0)', 'VJ: sF = 1.025, mN = 5, mS = (15,15)', 'VJ: sF = 1.025, mN = 5, mS = (30,30)', 'VJ: sF = 1.025, mN = 5, mS = (45,45)'])

