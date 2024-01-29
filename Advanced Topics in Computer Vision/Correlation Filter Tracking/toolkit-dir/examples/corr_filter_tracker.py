import numpy as np
import cv2
import matplotlib.pyplot as plt
from ex3_utils import create_gauss_peak, create_cosine_window
from ex2_utils import get_patch
from utils.tracker import Tracker
from numpy.fft import fft2, ifft2


class CorrFilterTracker(Tracker):

    def __init__(self, enlarge_factor, gaussian_sigma, filter_lambda, update_factor):
        self.enlarge_factor = enlarge_factor
        self.gaussian_sigma = gaussian_sigma
        self.filter_lambda = filter_lambda
        self.update_factor = update_factor

    def name(self):
        return "Correlation"

    def initialize(self, image, region):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if len(region) == 8:
            x = np.array(region[::2])
            y = np.array(region[1::2])
            region = [np.min(x), np.min(y), np.max(x) - np.min(x) + 1, np.max(y) - np.min(y) + 1]

        self.window = max(region[2], region[3]) * self.enlarge_factor
        
        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        
        self.gaussian_peak = create_gauss_peak((int(self.window), int(self.window)), self.gaussian_sigma)
        gaussian_peak_size = np.shape(self.gaussian_peak)
        self.cosine_window = create_cosine_window(gaussian_peak_size)
        patch, _ = get_patch(image, self.position, gaussian_peak_size)
        patch = patch*self.cosine_window
        
        divident = fft2(self.gaussian_peak)* np.conjugate(fft2(patch))
        divisor = self.filter_lambda + fft2(patch)*np.conjugate(fft2(patch))
        self.filter_fft = divident/divisor
        

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)
        
        gaussian_peak_size = np.shape(self.gaussian_peak)
        patch, _ = get_patch(image, self.position, gaussian_peak_size)
        patch = patch*self.cosine_window
        patch_size = np.shape(patch)
        response = ifft2(self.filter_fft*fft2(patch))
        
        max_index = response.argmax()
        y_max, x_max = np.unravel_index(max_index, np.shape(response))

        if x_max > patch_size[0] / 2:
            x_max = x_max - patch_size[0]
        if y_max > patch_size[1] / 2:
            y_max = y_max - patch_size[1]

        x_new = self.position[0] + x_max
        y_new = self.position[1] + y_max

        self.position = (x_new, y_new)

        # Update model
        patch, _ = get_patch(image, self.position, gaussian_peak_size)
        patch = patch*self.cosine_window
        divident = fft2(self.gaussian_peak)* np.conjugate(fft2(patch))
        divisor = self.filter_lambda + fft2(patch)*np.conjugate(fft2(patch))
        filter_new = divident/divisor
        self.filter_fft = (1 - self.update_factor) * self.filter_fft + self.update_factor * filter_new

        return [x_new - (self.size[0] / 2), y_new - (self.size[1] / 2), self.size[0], self.size[1]]
        