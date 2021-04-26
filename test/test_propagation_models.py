import unittest
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.propagation_models import fresnel_propagation
from src.propagation_models import fresnel_propagation_bis
from src.propagation_models import fresnel_propagation_goodman

import astropy.units as u
import pyoptica as po


class TestPropagationModels(unittest.TestCase):

    def test_fresnel_propagation(self):
        # parameters
        wave_field_start = img=cv2.imread('square.jpg', 0)
        #print("Shape: {}".format(wave_field_start.shape))
        wavelength = 1 # not working for less than 1
        propagation_distance = 40
        target_shape = wave_field_start.shape
        sampling_rates = (1,1)

        # propagate wave
        wave_field_end = fresnel_propagation(wave_field_start, wavelength, propagation_distance, sampling_rates=sampling_rates)
        wave_field_end = np.abs(wave_field_end) ** 2  # get intensity

        # plot result
        plt.figure(figsize=(16, 9))
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Initial wave field'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(wave_field_end, cmap='gray')
        plt.title('Propagated wave field'), plt.xticks([]), plt.yticks([])
        plt.show()

    def test_fresnel_propagation_bis(self):
        # note: shape of matrix is the number of samples!
        # parameters
        wave_field_start = img = cv2.imread('square.jpg', 0)
        # print("Shape: {}".format(wave_field_start.shape))
        wavelength = 523e-9  # nano-meters
        propagation_distance = 0.2  # unknown: units?
        target_shape = wave_field_start.shape
        sampling_rates = (10e-6, 10e-6)  # sampling every 10 micro-meters

        # propagate wave
        wave_field_end = fresnel_propagation_bis(wave_field_start, wavelength, propagation_distance, target_shape, sampling_rates)
        wave_field_end = np.abs(wave_field_end) ** 2  # get intensity

        # plot result
        plt.figure(figsize=(16, 9))
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Initial wave field'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(wave_field_end, cmap='gray')
        plt.title('Propagated wave field'), plt.xticks([]), plt.yticks([])
        plt.show()

    def test_fresnel_propagation_goodman(self):
        # parameters
        wave_field_start = img=cv2.imread('square.jpg', 0)
        #print("Shape: {}".format(wave_field_start.shape))
        wavelength = 1  # unknown: units?
        propagation_distance = 40  # unknown: units?
        target_shape = wave_field_start.shape

        # propagate wave
        wave_field_end = fresnel_propagation_goodman(wave_field_start, wavelength, propagation_distance, target_shape)

        # plot result
        wave_field_end_intensity = np.abs(wave_field_end) ** 2  # get intensity
        plt.figure(figsize=(16, 9))
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Initial wave field'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(wave_field_end_intensity, cmap='gray')
        plt.title('Propagated wave field'), plt.xticks([]), plt.yticks([])
        plt.show()


    def test_pyoptica_propagation(self):
        wavelength = 500 * u.nm
        pixel_scale = 22 * u.um
        npix = 1024

        w = 6 * u.mm
        h = 3 * u.mm

        wf = po.Wavefront(wavelength, pixel_scale, npix)
        ap = po.RectangularAperture(w, h)
        wf = wf * ap
        wf.plot(intensity='default')
        plt.show()

        f = 50 * u.cm
        fs_f = po.FreeSpace(f)
        wf_forward = wf * fs_f
        wf_forward.plot(intensity='default')
        plt.show()

    def test_inverse_propagation(self):
        # parameters
        wave_field_start = img=cv2.imread('square.jpg', 0)
        #print("Shape: {}".format(wave_field_start.shape))
        wavelength = 1  # unknown: units?
        propagation_distance = 40  # unknown: units?

        # propagate wave
        wave_field_end = fresnel_propagation(wave_field_start, wavelength, propagation_distance)
        wave_field_end_intensity = np.abs(wave_field_end) ** 2

        # inverse propagate back to original wave
        wave_field_start1 = fresnel_propagation(wave_field_end, wavelength, -propagation_distance)
        wave_field_start1_intensity = np.abs(wave_field_start1) ** 2

        np.testing.assert_array_almost_equal(wave_field_start, wave_field_start1_intensity)

        # plot result
        plt.figure(figsize=(16, 9))
        plt.subplot(121), plt.imshow(wave_field_end_intensity, cmap='gray')
        plt.title('Propagated wave field'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(wave_field_start1_intensity, cmap='gray')
        plt.title('Reconstructed original wave field'), plt.xticks([]), plt.yticks([])
        plt.show()

    def test_inverse_propagation_goodman(self):
        # parameters
        wave_field_start = img = cv2.imread('square.jpg', 0)
        # print("Shape: {}".format(wave_field_start.shape))
        wavelength = 1  # unknown: units?
        propagation_distance = 40  # unknown: units?
        target_shape = wave_field_start.shape

        # propagate wave
        wave_field_end = fresnel_propagation_goodman(wave_field_start, wavelength, propagation_distance, target_shape)
        wave_field_end_intensity = np.abs(wave_field_end) ** 2  # complex norm

        # inverse propagate back to original wave
        wave_field_start1 = fresnel_propagation_goodman(wave_field_end, wavelength, -propagation_distance, target_shape)
        wave_field_start1_intensity = np.abs(wave_field_start1) ** 2

        #np.testing.assert_array_almost_equal(wave_field_start, wave_field_start1_intensity)

        # plot result
        plt.figure(figsize=(16, 9))
        plt.subplot(121), plt.imshow(wave_field_end_intensity, cmap='gray')
        plt.title('Propagated wave field'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(wave_field_start1_intensity, cmap='gray')
        plt.title('Reconstructed original wave field'), plt.xticks([]), plt.yticks([])
        plt.show()