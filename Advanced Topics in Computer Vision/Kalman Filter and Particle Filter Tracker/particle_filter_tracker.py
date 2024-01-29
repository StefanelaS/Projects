# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 07:30:09 2023

@author: Asus
"""

import numpy as np
import cv2
from kalman_filter import get_symbolic_matrices, get_matrices_values
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram, Tracker
from ex4_utils import sample_gauss
from gensim.matutils import hellinger
import random

random.seed(1)

class PFTracker(Tracker):

    def initialize(self, image, region):
       
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
  
        self.kernel = create_epanechnik_kernel(region[2], region[3], self.parameters.sigma)
        kernel_size = np.shape(self.kernel)
        self.size = kernel_size
        self.patch, _ = get_patch(image, self.position, (kernel_size[1], kernel_size[0]))
        self.hist = extract_histogram(self.patch, self.parameters.nbins, weights=self.kernel)
        
        self.A, Q, _ = get_symbolic_matrices(self.parameters.model)
        self.Q, _ = get_matrices_values(1, self.parameters.q, Q)
        
        state = [self.position[0], self.position[1]]
        if self.parameters.model == 'NCV':
            state = [self.position[0], self.position[1], 0, 0]
        if self.parameters.model == 'NCA':
            state = [self.position[0], self.position[1], 0, 0, 0 ,0]
 
        self.particles = sample_gauss(state, self.Q, self.parameters.n_particles)
        self.weights = np.ones(self.particles.shape[0])
      
        
    def track(self, image):
        
        weights_norm =  self.weights/ np.sum(self.weights)  
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.parameters.n_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        particles_new = self.particles[sampled_idxs.flatten(), :]
        
        noise = sample_gauss(np.zeros(self.A.shape[0]), self.Q, self.parameters.n_particles)
        
        positions_x = np.zeros(self.parameters.n_particles)
        positions_y = np.zeros(self.parameters.n_particles)
        #print(particles_new)
        
        for i, particle in enumerate(particles_new):
      
            state_new = self.A @ particle + noise[i]
            particle = state_new
            positions = (state_new[0], state_new[1])
            patch, _ = get_patch(image, positions, (self.size[1], self.size[0])) #extract patch
            hist_p = extract_histogram(patch, self.parameters.nbins, weights=self.kernel) #extract histogram on patch
            hellinger_dist = hellinger(hist_p, self.hist) #calculate hellinger for hists with gensim
            self.weights[i] = np.exp(-0.5 * hellinger_dist**2 / 1**2)
            positions_x[i] = positions[0]
            positions_y[i] = positions[1]
        
        sum_weights = np.sum(self.weights)
        # if sum_weights == 0 :
        #     sum_weights = 1e-100
        
        if sum_weights == 0:
            x_new, y_new = self.position
            self.weights = np.ones(self.particles.shape[0])
           
        else:
            self.weights = self.weights / sum_weights
            
            # update position
            x_new = np.sum(np.multiply(self.weights, positions_x)) / np.sum(self.weights)
            y_new = np.sum(np.multiply(self.weights, positions_y)) / np.sum(self.weights)
            self.position = (x_new, y_new)
        
            # update histogram
            self.patch, _ = get_patch(image, self.position, (self.size[1], self.size[0]))
            alpha = self.parameters.alpha
            self.hist = (1 - alpha) * self.hist + alpha * extract_histogram(self.patch, self.parameters.nbins, weights=self.kernel)
        
        return [x_new - self.size[1] / 2, y_new - self.size[0] / 2, self.size[1], self.size[0]]


class PFTParams():
    def __init__(self, model, n_particles = 10, q = 100):
        self.model = model
        self.n_particles = n_particles
        self.q = q
        self.enlarge_factor = 2
        self.sigma = 0.1
        self.nbins = 16
        self.alpha = 0.2