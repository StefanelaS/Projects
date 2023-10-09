# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 23:05:01 2022

@author: Asus
"""

import glob
import os
from PIL import Image
from scipy.spatial.distance import euclidean, cosine
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


#LOADING THE IMAGES

"""
All loaded images are stored in a matrix of size 100x10,
(where 100 is corresponding to the number of classes and 10 is corresponding to
the number of images in each class).

"""

images = []
pix_maps = []
ft_vectors = []


for i in range(1,101):
    
    # Get the current working directory
    current_directory = os.getcwd()

    # Construct the full path to the data file
    img_dir = os.path.join(current_directory, 'awe')
    
    if i<10:
        path=os.path.join(img_dir, '00{}'.format(i),'*png')
    elif i==100:
        path=os.path.join(img_dir, '{}'.format(i),'*png')
    else:
        path=os.path.join(img_dir, '0{}'.format(i),'*png')  
    
    img_class = []
    pix_map_class = []
    ft_vectors_class = []

    
    for file in glob.glob(path):
        input_img = Image.open(file, 'r')
        input_img = input_img.convert('L')
        input_img = input_img.resize((128,128))   # choose the image size
        pix_arr = np.array(input_img.getdata())
        pix_map = np.reshape(pix_arr,(128,128))   # choose the image size
        ft_vectors_class.append(pix_arr)
        img_class.append(input_img)
        pix_map_class.append(pix_map)
       
    images.append(img_class)
    pix_maps.append(pix_map_class)             # matrix 100x10 where each cell contains pixel map matrix of one image
    ft_vectors.append(ft_vectors_class)        # matrix 100x10 where each cell contains feature vector of one image

plt.imshow(images[0][0], cmap='gray')
plt.axis('off')


# MINIMUM DISTANCE AND RANK 1 RECOGNITION RATE CALCULATION FUNCTIONS

"""
Function min_distance takes as an input following parametars: 
    features - database of feature vectors in the form of matrix, 
    folder - ordinary number of the row in features,
    file - ordinary number of the column in features ,
    method - method for the distance calculation (euclidean, cosine...).
This function compares feature vector at the position (folder,file) with all 
other vectors from the features, finds the vector with the minimum distance by
specified method and returns 1 if the two vectors belong to the same class(person).

"""

def min_distance(features, folder, file, method):
    min_dist = 1000000
    best_match_class = 0
    file_num = 0
    rank_1_match = 0
    for i in range(0,100):
        for j in range(0,10):
            if (i == folder) and (j == file):
                continue
            dist = method(features[folder][file], features[i][j])
            if dist < min_dist:
                min_dist = dist
                best_match_class = i
                file_num = j         
    if best_match_class == folder:
        rank_1_match = 1     
    return rank_1_match


"""
Function recognition_rate takes as an input following parametars:
    features - database of feature vectors in the form of matrix 
    method - method for the distance calculation (e.g. euclidean, cosine...).
This function calculates and returns rank-1 recognition rate.
"""
def recognition_rate (features, method):
    rank_1_match_sum = 0
    total = 0
    for i in range(0,100):
        for j in range(0,10):
            rank_1_match_sum = rank_1_match_sum + min_distance(features, i, j, method)
            total = total + 1
    return (rank_1_match_sum/total)*100


# LOCAL BINARY PATTERN (LBP)

"""
Function recognition_rate takes as an input following parametars:
    img - image in a form of pixel map,
    L - number of neighbors,
    r - radius,
    step - step size at which pixels for LBP code calculating are taken.
This function returns an image obtained after LBP procedure.

"""

def lbp (img, L, r, step):
    
    y = len(img[0][:])
    x = len(img[:][0])
    zero_one = np.zeros((x,y))
    lbp_val = np.zeros((x,y))
        
    for i in range(r, x-r, step):
        for j in range(r, y-r, step):
            
            if img[i][j]>img[i][j-r]:
                zero_one[i][j-r] = 0
            else:
                zero_one[i][j-r] = 1
            
            if img[i][j]>img[i][j+r]:
                zero_one[i][j+r] = 0
            else:
                zero_one[i][j+r] = 1
                
            if img[i][j]>img[i-r][j]:
                zero_one[i-r][j] = 0
            else:
                zero_one[i-r][j] = 1
                 
            if img[i][j]>img[i+r][j]:
                zero_one[i+r][j] = 0
            else:
                zero_one[i+r][j] = 1
            
            if L == 8 :
                if img[i][j]>img[i-r][j-r]:
                    zero_one[i-r][j-r] = 0
                else:
                    zero_one[i-r][j-r] = 1  
                
                if img[i][j]>img[i-r][j+r]:
                    zero_one[i-r][j+r] = 0
                else:
                    zero_one[i-r][j+r] = 1
                
                if img[i][j]>img[i+r][j-r]:
                    zero_one[i+r][j-r] = 0
                else:
                    zero_one[i+r][j-r] = 1
            
                if img[i][j]>img[i+r][j+r]:
                    zero_one[i+r][j+r] = 0
                else:
                    zero_one[i+r][j+r] = 1              
                    
            if L == 8:
                lbp_val[i][j] = zero_one[i-r][j-r]*128 + zero_one[i-r][j]*64 + zero_one[i-r][j+r]*32 + zero_one[i][j+r]*16 
                lbp_val[i][j] = lbp_val[i][j] +zero_one[i+r][j+r]*8 + zero_one[i+r][j]*4 + zero_one[i+r][j-r]*2 + zero_one[i][j-r]*1
            else:
                lbp_val[i][j] = zero_one[i-r][j]*8 + zero_one[i][j+r]*4 + zero_one[i+r][j]*2 + zero_one[i][j-r]*1
                
    if step > r:
        new_val = np.zeros((int(x/step),int(y/step)))
        for i in range(r, x-r, step):
            for j in range(r, y-r, step):
                new_val[int((i-r)/step)][int((j-r)/step)] = lbp_val[i][j]     
        lbp_val = new_val
            
    return lbp_val


images_lbp = []
ft_vectors_lbp = []

for i in range(0,100):
    img_class_lbp = []
    ft_class_lbp = []
    
    for j in range(0,10):
        image_lbp = lbp (pix_maps[i][j], 8, 1, 1)      # choose code length, radius and step size
        ft_arr_lbp = image_lbp.flatten()
        img_class_lbp.append(image_lbp)
        ft_class_lbp.append(ft_arr_lbp)
        
    images_lbp.append(img_class_lbp)                   # matrix 100x10 where each cell contains LBP pixel map of one image
    ft_vectors_lbp.append(ft_class_lbp)                # matrix 100x10 where each cell contains LBP feature vector of one image

plt.imshow(images_lbp[0][0], cmap='gray')
plt.axis('off')


# PLAIN PIXEL BY PIXEL - EUCLIDEAN
pbp_euc = recognition_rate(ft_vectors, euclidean)
print('PBP - recognition rate (euclidean):',pbp_euc)

# LBP - EUCLIDEAN
lbp_euc = recognition_rate(ft_vectors_lbp, euclidean)
print('Recognition rate (euclidean):',lbp_euc)

# LBP - COSINE
lbp_cos = recognition_rate(ft_vectors_lbp, cosine)
print('Recognition rate (euclidean):',lbp_cos)

# SKIMAGE IMPLEMENTATION
images_sk = [] 
ft_vectors_sk = []

for i in range(0,100):
    img_class_sk = []
    ft_class_sk = []

    for j in range(0,10):
        image_sk = local_binary_pattern (pix_maps[i][j], 8, 1, method='default')    # choose code length, radius and method
        ft_arr_sk = image_sk.flatten()
        img_class_sk.append(image_sk)            
        ft_class_sk.append(ft_arr_sk)
        
    images_sk.append(img_class_sk)                  # matrix 100x10 where each cell contains LBP pixel map of one image
    ft_vectors_sk.append(ft_class_sk)               # matrix 100x10 where each cell contains LBP feature vector of one image

plt.imshow(images_sk[0][0], cmap='gray')
plt.axis('off')

# SCIKIT LBP - EUCLIDEAN
sk_euc = recognition_rate(ft_vectors_sk, euclidean)
print('Recognition rate (euclidean):',sk_euc)


