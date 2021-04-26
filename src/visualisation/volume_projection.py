import numpy as np
from numpy.fft import fftshift, fft2
import matplotlib.pyplot as plt
import os

from src.propagation_models import fresnel_propagation
from src.visualisation.utils import remove_files, generate_images
from src.visualisation.video_generator import generate_video_from_focal_stack


class VolumeProjection:

    def __init__(self, image_shape, incident_field, hologram_patterns, start_pos, end_pos, wavelength):
        self.image_shape = image_shape
        self.incident_field = incident_field
        self.hologram_patterns = hologram_patterns
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.wavelength = wavelength

    def focal_stack(self, n_planes=100):
        plane_locations = np.linspace(self.start_pos, self.end_pos, num=n_planes)
        focal_stack = np.zeros(self.image_shape + (n_planes,))

        # compute projection for each focal plane by averaging projections from all holograms
        for plane_index in range(0, n_planes):
            projection_sequence = np.zeros(self.image_shape + (self.hologram_patterns.shape[2],), dtype=np.complex128)
            for hologram_index in range(0, self.hologram_patterns.shape[2]):
                DMD_binary_pattern = self.incident_field * np.exp(1j * (np.angle(self.hologram_patterns[:,:,hologram_index])))  # fixme add binarization
                projection_sequence[:, :, hologram_index] = \
                    fresnel_propagation(fftshift(fft2(DMD_binary_pattern)), self.wavelength, plane_locations[plane_index])  # fixme warning complex casting
            focal_stack[:, :, plane_index] = np.mean(np.abs(projection_sequence) ** 2, axis=2)  # get average light intensity

        return focal_stack

    def generate_video(self, file, n_planes=100):
        focal_stack = self.focal_stack(n_planes=n_planes)
        generate_video_from_focal_stack(focal_stack.astype(np.uint8), self.image_shape, file)

    def generate_images(self, path, n_planes=10):
        focal_stack = self.focal_stack(n_planes=n_planes)
        generate_images(focal_stack, path)





