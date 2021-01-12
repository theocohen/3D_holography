import unittest
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.propagation_models import fresnel_propagation

class TestPropagationModels(unittest.TestCase):
    def test_fresnel_propagation(self):
        # parameters
        wave_field_start = img=cv2.imread('square.jpg', 0)
        #print("Shape: {}".format(wave_field_start.shape))
        wavelength = 1  # unknown: units?
        propagation_distance = 40  # unknown: units?

        # propagate wave
        wave_field_end = fresnel_propagation(wave_field_start, wavelength, propagation_distance)
        wave_field_end = np.abs(wave_field_end)  # complex norm

        # plot result
        plt.figure(figsize=(16, 9))
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Initial wave field'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(wave_field_end, cmap='gray')
        plt.title('Propagated wave field'), plt.xticks([]), plt.yticks([])
        plt.show()