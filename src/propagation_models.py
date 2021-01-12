import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

def fresnel_propagation(wave_field_start, wavelength, propagation_distance):
    """
    Inspired from:
        * http://stackoverflow.com/questions/20971945/fresnel-diffraction-in-two-steps
        * https://github.com/dalerxli/Fresnel-Diffraction-1/blob/master/Fresnel%2Bdiffraction%2Bin%2Bpython.ipynb

    :param wave_field_start: 2D intensity wavefront at the start ('UO)
    :param wavelength: 'lambda' in optics
    :param propagation_distance: distance between the start and end plane ('z')
    :return: complex propagated wave field
    """

    wave_vector_norm = 2 * np.pi / wavelength  # k

    nx, ny = wave_field_start.shape  # shape of start 2D wavefront
    dx = dy = 1  # shape of end 2D wavefront
    scale_factor = dx / nx
    assert scale_factor == dy / ny  # scale factor should be equal to ny / dy

    # unknown: what do u and v represent? Guess: angular spectrum coordinates
    # unknown: why not symmetric? (error?)
    u = np.arange(-nx/2 + 1, nx/2 + 1, 1) * scale_factor
    v = np.arange(-ny/2 + 1, ny/2 + 1, 1) * scale_factor
    u, v = np.meshgrid(u, v)

    f = fftshift(fft2(wave_field_start))  # Fast Fourier Transform
    f_shift = fftshift(f)  # unknown: Why "Correct the shift in high and low frequencies"?

    # unknown: correct formula?
    propagator_matrix = np.exp(1j * wave_vector_norm * propagation_distance - np.pi * 1j * wavelength * propagation_distance * (u ** 2 + v ** 2))
    #propagator_matrix = np.exp(2 * np.pi * 1j * propagation_distance * scale_factor * np.sqrt((1/wavelength) ** 2 - u ** 2 - v ** 2)) # github source version

    return ifft2(ifftshift(f_shift * propagator_matrix))  # unknown: apply ifftshift?




    