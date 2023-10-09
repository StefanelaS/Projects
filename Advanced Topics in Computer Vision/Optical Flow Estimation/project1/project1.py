# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:04:24 2023

@author: Asus
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from ex1_utils import rotate_image , show_flow
from of_methods import lucas_kanade, horn_schunck

# LOADING ALL THE IMAGES
im1 = np.random.rand(200,200).astype(np.float32)
im2 = im1.copy( )
im2 = rotate_image(im2, -1)

im1_lab = cv2.imread('lab2/049.jpg', cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
im2_lab = cv2.imread('lab2/050.jpg', cv2.IMREAD_GRAYSCALE).astype('float32')/255.0
im1_off = cv2.imread('disparity/office_left.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
im2_off = cv2.imread('disparity/office_right.png', cv2.IMREAD_GRAYSCALE).astype('float32')/255.0
im1_car = cv2.imread('collision/00000148.jpg', cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
im2_car = cv2.imread('collision/00000149.jpg', cv2.IMREAD_GRAYSCALE).astype('float32')/255.0
images = [(im1_car, im2_car), (im1_lab, im2_lab), (im1_off, im2_off)]


# LK and HS on random noise images

U_lk, V_lk = lucas_kanade(im1, im2, 3)
U_hs, V_hs = horn_schunck(im1, im2, 0.5, 1000)

fig1, ((ax1_11 ,ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2, figsize = (5.3,5))
ax1_11.imshow(im1)
ax1_12.imshow(im2)
show_flow(U_lk, V_lk, ax1_21, type='angle')
show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
fig1.suptitle ('Lucas−Kanade Optical Flow')
#plt.savefig('random_LK.png', dpi=300)
plt.show()

fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2, figsize= (5.3,5))
ax2_11.imshow(im1)
ax2_12.imshow(im2)
show_flow(U_hs, V_hs, ax2_21, type='angle')
show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
fig2.suptitle('Horn−Schunck Optical Flow')
#plt.savefig('random_HS.png', dpi=300)
plt.show()

def test_on_3_pairs():
    index = 1
    for im1, im2 in images:
        U_lk, V_lk = lucas_kanade(im1, im2, 5)
        U_hs, V_hs = horn_schunck(im1, im2, 0.5, 1000)
         
        # Creating a plot
        fig, ((ax_1, ax_2, ax_3, ax_4, ax_5, ax_6)) = plt.subplots(1, 6, figsize = (22,3))
        ax_1.imshow(im1, cmap='gray')
        ax_2.imshow(im2, cmap ='gray')
        show_flow(U_lk, V_lk, ax_3, type='angle')
        show_flow(U_lk, V_lk, ax_4, type='field', set_aspect=True)
        show_flow(U_hs, V_hs, ax_5, type='angle')
        show_flow(U_hs, V_hs, ax_6, type='field', set_aspect=True)
        ax_1.set_title('Input image 1')
        ax_2.set_title('Input image 2')
        ax_3.set_title('LK -Angle')
        ax_4.set_title('LK - Field')
        ax_5.set_title('HS - Angle')
        ax_6.set_title('HS - Field')
        
        #plt.savefig('image_pair{}.png'.format(index), dpi=300)
        index += 1
        plt.show( )

def LK_parameters():
    U_lk1, V_lk1 = lucas_kanade(im1_car, im2_car, N =15, sigma=1)
    U_lk2, V_lk2 = lucas_kanade(im1_car, im2_car, N = 15, sigma=3)
    U_lk3, V_lk3 = lucas_kanade(im1_car, im2_car, N = 15, sigma=5)
    U_lk4, V_lk4 = lucas_kanade(im1_car, im2_car, N = 3, sigma=1)
    U_lk5, V_lk5 = lucas_kanade(im1_car, im2_car, N = 10, sigma=1)
    U_lk6, V_lk6 = lucas_kanade(im1_car, im2_car, N = 25, sigma=1)
    fig, ((ax_11 ,ax_12, ax_13), (ax_21, ax_22, ax_23)) = plt.subplots(2, 3, figsize = (8,5))
    show_flow(U_lk1, V_lk1, ax_11, type='field', set_aspect=True)
    show_flow(U_lk2, V_lk2, ax_12, type='field', set_aspect=True)
    show_flow(U_lk3, V_lk3, ax_13, type='field', set_aspect=True)
    show_flow(U_lk4, V_lk4, ax_21, type='field', set_aspect=True)
    show_flow(U_lk5, V_lk5, ax_22, type='field', set_aspect=True)
    show_flow(U_lk6, V_lk6, ax_23, type='field', set_aspect=True)
    ax_11.set_title('N = 15, sigma=1')
    ax_12.set_title('N = 15, sigma=3')
    ax_13.set_title('N = 15, sigma=5')
    ax_21.set_title('N = 3, sigma=1')
    ax_22.set_title('N = 10, sigma=1')
    ax_23.set_title('N = 25, sigma=1')
    #plt.savefig('parameter_LK.png', dpi=300)
    plt.show()
    
def HS_parameters():
    U_hs1, V_hs1 = horn_schunck(im1_car, im2_car, 0.01, 1000)
    U_hs2, V_hs2 = horn_schunck(im1_car, im2_car, 0.1, 1000)
    U_hs3, V_hs3 = horn_schunck(im1_car, im2_car, 1, 1000)
    U_hs4, V_hs4 = horn_schunck(im1_car, im2_car, 0.5, 100)
    U_hs5, V_hs5 = horn_schunck(im1_car, im2_car, 0.5, 1000)
    U_hs6, V_hs6 = horn_schunck(im1_car, im2_car, 0.5, 10000)
    fig, ((ax_11 ,ax_12, ax_13), (ax_21, ax_22, ax_23)) = plt.subplots(2, 3, figsize = (9,5))
    show_flow(U_hs1, V_hs1, ax_11, type='field', set_aspect=True)
    show_flow(U_hs2, V_hs2, ax_12, type='field', set_aspect=True)
    show_flow(U_hs3, V_hs3, ax_13, type='field', set_aspect=True)
    show_flow(U_hs4, V_hs4, ax_21, type='field', set_aspect=True)
    show_flow(U_hs5, V_hs5, ax_22, type='field', set_aspect=True)
    show_flow(U_hs6, V_hs6, ax_23, type='field', set_aspect=True)
    ax_11.set_title('λ = 0.01, n_iter=1000')
    ax_12.set_title('λ = 0.1, n_iter=1000')
    ax_13.set_title('λ = 1, n_iter=1000')
    ax_21.set_title('λ = 0.5, n_iter=100')
    ax_22.set_title('λ = 0.5, n_iter=1000')
    ax_23.set_title('λ = 0.5, n_iter=10000')
    #plt.savefig('parameters_HS.png', dpi=300)
    plt.show()    
    
# Speed + Initializing HS with LK
def init_HS_with_LK():
    
    im1 = np.random.rand(200,200).astype(np.float32)
    im2 = im1.copy( )
    im2 = rotate_image(im2, -1)

    start = time.time()
    U_lk, V_lk = lucas_kanade(im1, im2, N=5, sigma=1)
    end = time.time()
    print('Runtime for LK on random noise:', end-start)
    start = time.time()
    U_hs, V_hs = horn_schunck(im1, im2, 0.5, 10000, init_LK=False)
    end = time.time()
    print('Runtime for HS:', end-start)
    start = time.time()
    U_hs_lk, V_hs_lk = horn_schunck(im1, im2, 0.5, 10000, init_LK=True)
    end = time.time()
    print('Runtime for HS initialized with LK:', end-start)
    start = time.time()
    U_lk2, V_lk2 = lucas_kanade(im1_car, im2_car, N=5, sigma=1)
    end = time.time()
    print('Runtime for LK on first pair of images:', end-start)
    start = time.time()
    U_hs2, V_hs2 = horn_schunck(im1_car, im2_car, 0.5, 10000, init_LK=False)
    end = time.time()
    print('Runtime for HS on first pair of images:', end-start)
    start = time.time()
    U_hs_lk2, V_hs_lk2 = horn_schunck(im1_car, im2_car, 0.5, 10000, init_LK=True)
    end = time.time()
    print('Runtime for HS initialized with LK:', end-start)
    
    fig, ((ax_11, ax_12)) = plt.subplots(1, 2, figsize= (7,3))
    show_flow(U_hs2, V_hs2, ax_11, type='field', set_aspect=True)
    show_flow(U_hs_lk2, V_hs_lk2, ax_12, type='field', set_aspect=True)
    ax_11.set_title('a)')
    ax_12.set_title('b)')
    #plt.savefig('HS_init_LK', dpi=300)
    plt.show()


test_on_3_pairs()
LK_parameters()
HS_parameters()
init_HS_with_LK()