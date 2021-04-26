from itertools import product

import numpy as np
from abc import ABCMeta, abstractmethod
from PIL import Image, ImageDraw, ImageFont

class TargetVolume(metaclass=ABCMeta):
    def __init__(self, shape, plane_locations):
        self.shape = shape
        self.volume = np.zeros(shape)
        self.plane_locations = plane_locations

    @property
    def point_cloud_arrays(self):
        """fixme TOO SLOW
        Converts 3D volume array into point cloud data representation used to visualise the volume
        :return: (x, y, z, scalars) with the 3 first arrays give the coordinates, and the last the scalar value
        """
        xx, yy, zz = zip(*[(x, y, self.plane_locations[z])
                           for (x, y, z) in product(range(self.shape[0]), range(self.shape[1]), range(self.shape[2]))])
        values = self.volume.flatten()
        return xx, yy, zz, values


class Sphere(TargetVolume):

    def __init__(self, shape, plane_locations, thickness=5, fill_factor=.9):
        super(Sphere, self).__init__(shape, plane_locations)
        xx, yy = np.mgrid[:shape[0], :shape[1]]  # coordinate mesh
        center = (shape[0] / 2, shape[1] / 2)  # center coordinates
        circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2  # distance from center
        for plane_index in range(0, shape[2]):
            radius = (shape[2] - 2 * np.abs(plane_index - shape[2] / 2)) / shape[2] * fill_factor * (np.min(shape[:2]) / 2)
            self.volume[:, :, plane_index] = (circle < (radius + thickness / 2) ** 2) & (circle > (radius - thickness / 2) ** 2)

class Digits(TargetVolume):

    def __init__(self, shape, plane_locations, font_path):
        super(Digits, self).__init__(shape, plane_locations)
        for digit in range(1, shape[2]):
            text = str(digit)
            img = Image.new('P', shape[:2], 0)
            font = ImageFont.truetype(font_path + "OpenSans-Regular.ttf", size=min(shape[:2]) // len(text))
            text_width, text_height = font.getsize(text)
            offset = ((shape[0] - text_width) // 2, ((shape[1] - text_height)))
            draw = ImageDraw.Draw(img)
            draw.text(offset, text, 1, font=font)
            self.volume[:, :, digit - 1] = np.asarray(img)