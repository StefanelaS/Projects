# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:40:58 2023

@author: Asus
"""
import numpy as np
import cv2
from ex1_utils import gaussderiv, gausssmooth


def lucas_kanade(im1, im2, N=5, sigma=1, return_grads = False):
      
      # Computing spacial and temporal derivatives
      Ix1, Iy1 = gaussderiv(im1, sigma)
      Ix2, Iy2 = gaussderiv(im2, sigma)
      Ix = (Ix1+Ix2)/2
      Iy = (Iy1+Iy2)/2
      It = im2 - im1
      It = gausssmooth(It, 1)
     
      kernel = np.ones((N,N), dtype=np.float32)

      # Computing the products of derivatives
      IxIx = cv2.filter2D(Ix * Ix, -1, kernel)
      IyIy = cv2.filter2D(Iy * Iy, -1, kernel)
      IxIy = cv2.filter2D(Ix * Iy, -1, kernel)
      IxIt = cv2.filter2D(Ix * It, -1, kernel)
      IyIt = cv2.filter2D(Iy * It, -1, kernel)
     
      D = (IxIx * IyIy - IxIy**2)
      correction = 1e-06
     
      U = np.where(D != 0.0, -(IyIy * IxIt - IxIy * IyIt) / (D + correction), 0.0)
      V = np.where(D != 0.0, -(IxIx * IyIt - IxIy * IxIt) / (D + correction), 0.0)
      
      if return_grads == True:
          return U, V, Ix, Iy, It, IxIx, IyIy
    
      return U, V


def horn_schunck(im1, im2, lmbd=0.5, num_iters=1000, sigma=1, eps=1e-4, init_LK=False):
    
    if init_LK == False:
        # Gaussian smoothing
        im1 = gausssmooth(im1, sigma)
        im2 = gausssmooth(im2, sigma)
        
        # Computing spacial and temporal derivatives
        kernel_x = np.array([[-1/2, 1/2], [-1/2, 1/2]])
        kernel_y = kernel_x.T
        kernel_t = np.array([[1/4, 1/4], [1/4, 1/4]])
        Ix = cv2.filter2D(im1, -1, kernel_x)
        Iy = cv2.filter2D(im1, -1, kernel_y)
        It = cv2.filter2D(im1 - im2, -1, kernel_t)
        
        kernel = np.ones((5,5), dtype=np.float32)
        # Computing the products of derivatives
        IxIx = cv2.filter2D(Ix * Ix, -1, kernel)
        IyIy = cv2.filter2D(Iy * Iy, -1, kernel)
        
        U = np.zeros_like(im1)
        V = np.zeros_like(im1)
        
        D= IxIx + IyIy + lmbd
        Ld = np.array([[0, 1/4, 0], [1/4, 0, 1/4], [0, 1/4, 0]])
    
    if init_LK == True:
        U, V, Ix, Iy, It, IxIx, IyIy = lucas_kanade(im1, im2, 5, return_grads=True)
        D = IxIx + IyIy + lmbd
        Ld = np.array([[0, 1/4, 0], [1/4, 0, 1/4], [0, 1/4, 0]])
    
    for i in range(num_iters):
        # Computing update for U and V
        U_a = cv2.filter2D(U, -1, Ld)
        V_a = cv2.filter2D(V, -1, Ld)
        P = Ix*U_a + Iy*V_a + It
        # Updating U and V
        U_new = U_a - Ix*(P/D)
        V_new = V_a - Iy*(P/D)
        # Checking convergence
        diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
        if diff < eps:
            break

        U = U_new
        V = V_new

    return U, V


