import unittest
import matplotlib.pyplot as plt
import numpy as np

from src.target_volume import Sphere, Digits
from src.visualisation.video_generator import generate_video_from_focal_stack, read_video


class TestVideoGenerator(unittest.TestCase):

    def test_volume_slice(self):
        shape = (512, 512 , 10)
        plane_locations = np.linspace(-300, 300, num=shape[2])  # unknown why negative
        target = Digits(shape, plane_locations, "./../../")

        file = '../../output/projection_test.avi'
        generate_video_from_focal_stack(target.volume, shape[:2], file)
        read_video(file, delay_per_frame=100)
