# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 01:29:23 2023

@author: Asus
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ex2_utils import generate_responses_1, generate_responses_2, get_patch
from ex2_utils import extract_histogram, backproject_histogram


def create_kernel(kernel_size):
    n_rows, n_cols = kernel_size

    if n_rows % 2 == 0:
        n_rows += 1
    if n_cols % 2 == 0:
        n_cols += 1
    
    center_row = (n_rows - 1) // 2
    center_col = (n_cols - 1) // 2
    
    kernel_X = np.arange(n_cols) - center_col
    kernel_X = np.tile(kernel_X, (n_rows, 1))

    kernel_Y = np.arange(n_rows) - center_row
    kernel_Y = np.tile(kernel_Y, (n_cols, 1)).T
    
    return kernel_X, kernel_Y


def mean_shift(image, starting_point, kernel_size, max_iterations=5000, tolerance=1e-2, hist=None, kernel=None, nbins=None):
    
    # Creating kernels
    kernel_X, kernel_Y = create_kernel(kernel_size)

    # Initializing window around the starting point
    window_center = np.array([starting_point[0], starting_point[1]])
    window, _ = get_patch(image, window_center, (kernel_size[1], kernel_size[0]))
    x = starting_point[0]
    y = starting_point[1]

    # Plotting the starting point
    # fig, ax = plt.subplots()
    # ax.imshow(image, cmap ='gray')
    # ax.add_patch(Circle(window_center, radius = 3, edgecolor='red', facecolor='red'))
    
    # Iterating until convergence or maximum number of iterations is reached
    for n_iter in range(max_iterations):
        
        if kernel is not None:
            q = extract_histogram(window, nbins, weights=kernel)
            v = np.sqrt(np.divide(hist, q + tolerance))
            w = backproject_histogram(window, v, nbins)
        else:
            w = window
        
        mean_shift_x = (np.sum(np.multiply(kernel_X, w)))/(np.sum(w))
        mean_shift_y = (np.sum(np.multiply(kernel_Y, w)))/(np.sum(w))
       
        if abs(mean_shift_x) < tolerance and abs(mean_shift_y) < tolerance:
            break
       
        x = x + mean_shift_x
        y = y + mean_shift_y
        
        window_center = np.array([x, y])
        window, _ = get_patch(image, window_center, (kernel_size[1], kernel_size[0]))
    
    # Plotting the ending point    
    # fig, ax = plt.subplots()
    # ax.imshow(image, cmap = 'gray')
    # ax.add_patch(Circle(window_center, radius = 3, edgecolor='red', facecolor='red'))
    
    coord_max = (int(x), int(y))
    value = image[int(y),int(x)] 
        
    return coord_max, value, n_iter+1



          
if __name__== "__main__":

     image = generate_responses_1()
     # Changing kernel size
     kernel_sizes = [(3,3), (5,5), (7,7), (9,9)]
     for i in range(0, len(kernel_sizes)):
         print('\nKERNEL SIZE:', kernel_sizes[i])
         start = time.time()
         coordinates, value, n_iter = mean_shift(image, (20,60), kernel_sizes[i])
         end = time.time()
         print('Coordinates of the maximum:', coordinates)
         print('Function value:', value)
         print('Number of iterations:', n_iter)
         print('Runtime:', end - start)

     # Starting from different points
     starting_points = [(20,60), (40,60), (80,20)]
     for i in range(0, len(starting_points)):
         print('\nSTARTING POINT:', starting_points[i])
         start = time.time()
         coordinates, value, n_iter = mean_shift(image, starting_points[i], (7,7))
         end = time.time()
         print('Coordinates of the maximum:', coordinates)
         print('Function value:', value)
         print('Number of iterations:', n_iter)
         print('Runtime:', end - start)


     # Changing termination criteria
     tolerances = [(1e-2), (5e-2), (1e-1)]
     for i in range(0, len(tolerances)):
         print('\nTERMINATION CRITERIA:', tolerances[i])
         start = time.time()
         coordinates, value, n_iter = mean_shift(image, (20,60), (7,7), tolerance = tolerances[i])
         end = time.time()
         print('Coordinates of the maximum:', coordinates)
         print('Function value:', value)
         print('Number of iterations:', n_iter)
         print('Runtime:', end - start)

     image = generate_responses_2()
     # Starting from different points
     starting_points = [(35,35), (70,70)]
     for i in range(0, len(starting_points)):
        print('\nSTARTING POINT:', starting_points[i])
        start = time.time()
        coordinates, value, n_iter = mean_shift(image, starting_points[i], (7,7))
        end = time.time()
        print('Coordinates of the maximum:', coordinates)
        print('Function value:', value)
        print('Number of iterations:', n_iter)
        print('Runtime:', end - start)
