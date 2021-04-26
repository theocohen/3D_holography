import unittest
import numpy as np

from src.hologram_generation import time_averaged_gs_3d_hologram_generation
from src.target_volume import Sphere, Digits
from src.visualisation.utils import generate_images
from src.visualisation.video_generator import read_video, generate_video_from_focal_stack
from src.visualisation.volume_projection import VolumeProjection

OUTPUT_PATH = './../output/'

class TestHologramGenerator(unittest.TestCase):

    def test_time_averaged_gs_3d_hologram_generation(self):
        wavelength = 1  # fixme
        shape = (1024, 1024, 10)
        time_averages = 10
        plane_locations = np.linspace(-300, 300, num=shape[2])  # unknown why negative?
        incident_field = np.ones(shape[:2], dtype=np.cdouble)  # light wave in projected onto the DMD chip. Flat phase (0 angle) unknown: why flat phase?
        #target = Digits(shape, plane_locations, "./../")
        target = Sphere(shape, plane_locations)

        # display target volume
        """
        generate_video_from_focal_stack(target.volume, shape[:2], OUTPUT_PATH + 'target_sphere.avi')
        read_video(file)
        """
        generate_images(target.volume, OUTPUT_PATH + 'images/target/')

        # hologram patterns generation
        hologram_patterns = time_averaged_gs_3d_hologram_generation(target.volume, plane_locations, incident_field, wavelength, hologram_stack_size=time_averages)

        # display hologram patterns
        generate_images(np.abs(hologram_patterns) ** 2, OUTPUT_PATH + 'images/holograms/intensity/')
        generate_images(np.real(hologram_patterns), OUTPUT_PATH + 'images/holograms/real/')
        generate_images(np.angle(hologram_patterns), OUTPUT_PATH + 'images/holograms/angle/')

        #projection_shape = (shape[0] * 2, shape[0] * 2)
        projection = VolumeProjection(shape[:2], incident_field, hologram_patterns, plane_locations[0], plane_locations[-1], wavelength)

        # save projection stack as images
        projection.generate_images(OUTPUT_PATH + 'images/projection/')

        # display projection stack as a video
        """
        projection.generate_video(OUTPUT_PATH + 'reconstructed_projected_sphere.avi')
        read_video(file)
        """
