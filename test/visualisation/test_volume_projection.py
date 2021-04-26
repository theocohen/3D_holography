import unittest
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab

class TestVisualisation(unittest.TestCase):

    @mlab.show
    def test_volume_slice(self):
        x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]

        scalars = x * x * 0.5 + y * y + z * z * 2.0

        mlab.volume_slice(scalars, plane_orientation='x_axes')
        mlab.start_recording()
