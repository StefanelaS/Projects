# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:44:17 2023

@author: Asus
"""
#%% LOADING THE LIBRARIES 

import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from scipy.spatial.distance import euclidean
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#%% FUNCTIONS

def get_crop_img():
    
    # loading the original images
    img_dir = "C:/Users/Asus/Desktop/biometry/assignment5/ibb-joint-data/images"
    images = []
    
    for i in range(1,622):
        
        if i < 10:
            path = os.path.join(img_dir, '00{}'.format(i),'*png')
        elif i < 100:
            path = os.path.join(img_dir, '0{}'.format(i),'*png')
        else:
            path = os.path.join(img_dir, '{}'.format(i),'*png')  
        
        images_folder = []
        
        for file in glob.glob(path):
            input_img = cv2.imread(file)                                       # reads an image
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)              # transforms in to RGB
            images_folder.append(gray_img) 
        images.append(images_folder)
    
        
    # get YOLO box
    location ="C:/Users/Asus/Desktop/biometry/assignment5/yolov5/runs/detect"
    yolo_box = []
        
    for j in range(1,622):
            
        yolo_labels = []
        yolo_box_folder = []
        txt_location = os.path.join(location,'exp{}'.format(j),'labels/*txt')
            
        for file in glob.glob(txt_location):
            f = open(file, 'r')                                                     # opens txt file 
            content = f.read()                                                      # reads txt file
            f.close()
            content = content.split()                                               # separates values and creates a list
            content = content[1:5]                                                  # takes only coordinates and width and heigh
            yolo_labels.append(content)
                   
        for i in range(0, len(yolo_labels)):
            if len(yolo_labels[i])==0:
                yolo_box_i = []
            else:
                yolo_box_i = [0, 0, 0, 0]
                img_height, img_width = images[j-1][i].shape[:2]                           # returns image size in pixels
                yolo_box_i[2] = round(float(yolo_labels[i][2])*img_width)                  # converts normalized width to width in pixels
                yolo_box_i[3] = round(float(yolo_labels[i][3])*img_height)                 # converts normalized height to height in pixels
                yolo_box_i[0] = round(float(yolo_labels[i][0])*img_width-yolo_box_i[2]/2)  # calculates x coordinate of top-left pixel
                yolo_box_i[1] = round(float(yolo_labels[i][1])*img_height-yolo_box_i[3]/2) # calculates y coordinate of top-left pixel
            yolo_box_folder.append(yolo_box_i)
        yolo_box.append(yolo_box_folder)
        
    # crop the images to yolo box size
    crop_images = []
        
    for i in range(0,len(images)):
        crop_folder = []
            
        for j in range (0, len(images[i])):
            x1 = yolo_box[i][j][0]
            x2 = yolo_box[i][j][0] + yolo_box[i][j][2]
            y1 = yolo_box[i][j][1]
            y2 = yolo_box[i][j][1] + yolo_box[i][j][3]
            crop = images[i][j][y1:y2, x1:x2]
            crop_folder.append(crop)
        crop_images.append(crop_folder)
        
    return crop_images, yolo_box 

def ground_truth_box():
    
    masks_dir = "C:/Users/Asus/Desktop/biometry/assignment5/ibb-joint-data/masks"
    gt_box = []
    
    for i in range(1,622):
        
        if i < 10:
            path = os.path.join(masks_dir, '00{}'.format(i),'*png')
        elif i < 100:
            path = os.path.join(masks_dir, '0{}'.format(i),'*png')
        else:
            path = os.path.join(masks_dir, '{}'.format(i),'*png')  
            
        gt_box_folder = []
        
        for file in glob.glob(path):
            mask = cv2.imread(file)                                             # reads a mask
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)                       # transforms in to grayscale 
            
            # get rectangular coordinates
            x1, y1, w, h = cv2.boundingRect(gray)
            gt_box_folder.append([x1, y1, w, h])
        
        gt_box.append(gt_box_folder)
        
    return gt_box


def intersection_over_union(boxA, boxB):
    
    iou = []
    
    if len(boxA) == 0: 
        iou = 'no folder'
    elif len(boxB) == 0:
        iou = 0
    else:
        for i in range(0, len(boxB)):
            Ax1 = boxA[0]
            Ay1 = boxA[1]
            Aw = boxA[2]
            Ah = boxA[3]
            
            # finding x and y coordinates of bottom right pixel in ground truth box
            Ax2 = Ax1 + Aw
            Ay2 = Ay1 + Ah
            
            Bx1 = boxB[0]
            By1 = boxB[1]
            Bw = boxB[2]
            Bh = boxB[3]
            # finding x and y coordinates of bottom right pixel in detection box
            Bx2 = Bx1 + Bw
            By2 = By1 + Bh
            
        # calculates lenghts of overlapping rectangle sides
        delta_x = abs(max(Ax1, Bx1) - min(Ax2, Bx2))
        delta_y = abs(max(Ay1, By1) - min(Ay2, By2))
    
        # calculates area of intersection
        overlap = delta_x * delta_y
            
        # calculates IoU
        try:
            union = Aw * Ah + Bw * Bh - overlap
            iou = overlap / union
        except ZeroDivisionError:
            iou = 0
            
    return iou


def accuracy(iou):
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ACC = []

    for threshold in thresholds:
        TP = 0
        FP = 0
        FN = 0
        
        for value in iou :
            if value == 'no folder':
                break
            elif value == 0:                                                    # if there is no detection in an image it is considered false negative
                FN += 1                                                             
            else:
                if value >= threshold and value < 1:                            # checks if IoU value is higher than a threshold
                    TP += 1                                                     # if value is higher or equal count it as TP
                else:
                    FP += 1                                                     # if value is lower count it as FP 
        acc_thres = TP/(TP+FP+FN)                                               # calculates accuracy
        ACC.append(acc_thres)                                
    
    return  ACC, thresholds  

#%%

images, yolo_box = get_crop_img()
gt_box = ground_truth_box()

#%% YOLO accuracy and visualization 

# calculates IoU for yolo detection boxes
iou = []
for i in range (0,len(yolo_box)):
    for j in range(0,len(gt_box[i])):
        iou_i = intersection_over_union(gt_box[i][j], yolo_box[i][j])
        iou.append(iou_i)

acc, threshold = accuracy(iou)

plt.style.use('ggplot')
plt.figure()
plt.plot(threshold, acc)
plt.xticks(threshold)
plt.xlabel('Threshold Value')
plt.ylabel('Accuracy')
plt.show()

plt.figure()
plt.hist(iou, range = [0,1], bins = 50)
plt.xlabel('Intersection Over Union')
plt.ylabel('Number of Detections')
plt.show()

#%% Function for LBP and LBP accuracy calculation

def LBP(crop_img):

    images_lbp = [] 
    ft_vectors = []

    for i in range(0,len(crop_img)):
        images_lbp_folder = []
        ft_lbp_folder = []

        for j in range(0,len(crop_img[i])):
            resized = cv2.resize(crop_img[i][j],(32,32))
            lbp = local_binary_pattern(resized, 8, 2)         # choose code length, radius 
            ft_lbp = lbp.flatten()
            images_lbp_folder.append(lbp)            
            ft_lbp_folder.append(ft_lbp)
        
        images_lbp.append(images_lbp_folder)                  
        ft_vectors.append(ft_lbp_folder)                      
    
    return images_lbp, ft_vectors



def rank_1_accuracy(features):
    
    rank_1_match_sum = 0
    total = 0
    
    for i in range(0,len(features)):
        for j in range(0,len(features[i])):
            rank_1_match_sum = rank_1_match_sum + min_distance(features, i, j)
            total = total + 1
        print(rank_1_match_sum)
    return (rank_1_match_sum/total)*100



def min_distance(features, folder, file):
   
    min_dist = 1000000
    best_match_class = 0
    #file_num = 0
    rank_1_match = 0
   
    for i in range(0,len(features)):
        for j in range(0,len(features[i])):
            if (i == folder) and (j == file):
                continue
            else:
                dist = euclidean(features[folder][file], features[i][j])
            if dist < min_dist:
                min_dist = dist
                best_match_class = i
    if best_match_class == folder:
        rank_1_match = 1     
    
    return rank_1_match

#%% LBP

images_lbp, ft_vectors = LBP(images)
lbp_rank_1 = rank_1_accuracy(ft_vectors)
print('Recognition rate (euclidean):', lbp_rank_1)

#%% SVC on LBP features

y = []
for i in range(0,len(images)):
    if len(images[i]) == 0:
        pass
    else:
        for j in range(0,len(images[i])):
            y.append(i)

le = LabelEncoder()
y = le.fit_transform(y)

X = []
for i in range(0,len(ft_vectors)):
    if len(ft_vectors[i]) == 0:
        pass
    else:
        for j in range(0,len(ft_vectors[i])):
            norm = cv2.normalize(ft_vectors[i][j], None, 0, 1.0, cv2.NORM_MINMAX)
            norm = np.array(norm)
            norm = np.transpose(norm)
            X.append(norm[0])  

X1 = X
y1 = np.array(y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, stratify = y1, random_state = 1)

# Create an SVM classifier
clf = SVC(kernel='linear')

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy_SVC = accuracy_score(y_test, y_pred)

print('Accurancy SVC:', accuracy_SVC)

#%% SVC on original cropped image

X_org = []
for i in range(0,len(images)):
    if len(images[i]) == 0:
        pass
    else:
        for j in range(0,len(images[i])):
            img = cv2.normalize(images[i][j], None, 0, 1.0, cv2.NORM_MINMAX)
            img = resize(img,(64, 64))
            img = np.array(img)
            img = np.transpose(img)
            X_org.append(norm[0]) 

X2 = X_org
y2 = np.array(y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, stratify = y2, random_state = 1)

# Create an SVM classifier
clf = SVC(kernel='linear')

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy_SVC_or = accuracy_score(y_test, y_pred)

print('Accurancy SVC:', accuracy_SVC_or)

