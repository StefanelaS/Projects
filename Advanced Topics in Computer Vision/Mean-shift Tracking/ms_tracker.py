# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 04:04:06 2023

@author: Asus
"""
import numpy as np
from mean_shift import mean_shift
from ex2_utils import  get_patch, create_epanechnik_kernel, extract_histogram
from ex2_utils import Tracker


class MeanShiftTracker(Tracker):
    
    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
       
        self.kernel = create_epanechnik_kernel(region[2], region[3], self.parameters.sigma)
        kernel_size = np.shape(self.kernel)
        self.size = kernel_size
        self.patch, _ = get_patch(image, self.position, (kernel_size[1], kernel_size[0]))
        self.hist = extract_histogram(self.patch, self.parameters.nbins, weights=self.kernel)
    
    def track(self, image):
        
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)
        
        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        coordinates, _, _ = mean_shift(image, self.position, self.size, tolerance=self.parameters.tolerance,
                                       hist=self.hist, kernel=self.kernel, nbins=self.parameters.nbins)

        self.patch, _ = get_patch(image, coordinates, (self.size[1], self.size[0]))
        alpha = self.parameters.alpha
        self.hist = (1 - alpha) * self.hist + alpha * extract_histogram(self.patch, self.parameters.nbins, weights=self.kernel)
        self.position = coordinates
        return [coordinates[0], coordinates[1], self.size[1], self.size[0]]
        
    
class MSParams():
    def __init__(self, sigma, tolerance, nbins, alpha):
        self.enlarge_factor = 2
        self.sigma = sigma
        self.tolerance = tolerance
        self.nbins = nbins
        self.alpha = alpha