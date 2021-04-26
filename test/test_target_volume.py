import unittest
import numpy as np
from mayavi import mlab

from src.target_volume import Sphere


class TestTargetVolume(unittest.TestCase):

    @mlab.show
    def test_sphere_volume(self):
        shape = (1024, 1024, 10)
        plane_locations = np.linspace(-300, 300, num=shape[2])  # unknown why negative
        sphere = Sphere(shape, plane_locations)

        #mlab.volume_slice(sphere.point_cloud_arrays, plane_orientation='z_axes') too slow
        mlab.volume_slice(sphere.volume, plane_orientation='z_axes')
        mlab.start_recording()