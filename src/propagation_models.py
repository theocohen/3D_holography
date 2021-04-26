import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

def fresnel_propagation(wave_field_start, wavelength, z, sampling_rates=(1,1)):
    """
    Inspired from:
        * http://stackoverflow.com/questions/20971945/fresnel-diffraction-in-two-steps
        * https://github.com/dalerxli/Fresnel-Diffraction-1/blob/master/Fresnel%2Bdiffraction%2Bin%2Bpython.ipynb

    :param wave_field_start: 2D intensity wavefront at the start ('UO)
    :param wavelength: 'lambda' in optics
    :param z: propagation distance between the start and end plane ('z')
    :return: complex propagated wave field
    """

    k = 2 * np.pi / wavelength  # angular wave number

    nx, ny = wave_field_start.shape  # shape of start 2D wavefront
    dx, dy = sampling_rates

    delta_x = 1 / (nx * dx)  # fixme shouln't it be lambda * z / (L_x + L_eta) where L_x = L_eta = nx * dx ?
    delta_y = 1 / (ny * dy)

    u = np.arange(-nx/2 + 1, nx/2 + 1, 1) * delta_x
    v = np.arange(-ny/2 + 1, ny/2 + 1, 1) * delta_y
    u, v = np.meshgrid(u, v)

    G = fftshift(fft2(wave_field_start))  # why fftshift?

    H = np.exp(1j * k * z - np.pi * 1j * wavelength * z * (u ** 2 + v ** 2))  # fourier transformed Fresnel convolution kernel.
    #H = np.exp(2 * np.pi * 1j * z * scale_factor * np.sqrt((1/wavelength) ** 2 - u ** 2 - v ** 2)) # angular spectrum github source version

    return ifft2(ifftshift(G * H))  # ifftshift not necessary here


def fresnel_propagation_bis(wave_field_start, wavelength, z, target_shape, sampling_rates):
    """Attempt from CDGH course -> all blank"""
    k = 2 * np.pi / wavelength  # angular wave number

    Mx, My = wave_field_start.shape  # shape of start 2D wavefront
    Nx, Ny = target_shape # shape of end 2D wavefront

    Cx = Mx + Nx - 1  # -1 not necessary
    Cy = My + Ny - 1

    delta_x, delta_y = sampling_rates

    padS = np.zeros((Cx, Cy))
    padS[:Mx, :My] = wave_field_start
    G = fftshift(fft2(padS))

    u = np.arange(round(-Cx/2) + 1, round(Cx/2), 1) * delta_x
    v = np.arange(round(-Cy/2) + 1, round(Cy/2), 1) * delta_y
    u, v = np.meshgrid(u, v)

    #h = - 1 / (1j * wavelength * z) * np.exp(1j * k * z + 1j * np.pi / (wavelength * z) * (u ** 2 + v ** 2))  # Fresnel convolution kernel fixme shouldn't have - sign (?)
    # H = fftshift(fft2(h))
    H = np.exp(1j * k * z - np.pi * 1j * wavelength * z * (u ** 2 + v ** 2))  # fourier transformed Fresnel convolution kernel.

    tmp = ifft2(ifftshift(G * H))
    return tmp[:Nx, :Ny]

def fresnel_propagation_goodman(wave_field_start, wavelength, z, target_shape):
    """From Goodman chap 9.9"""
    k = 2 * np.pi / wavelength  # angular wave number

    L_xi, L_eta = wave_field_start.shape  # shape of start 2D wavefront
    L_x, L_y = target_shape # shape of end 2D wavefront

    B_x = (L_xi + L_x) / (2 * wavelength * z)
    B_y = (L_eta + L_y) / (2 * wavelength * z)

    delta_x = 1 / (2 * B_x)
    delta_y = 1 / (2 * B_y)

    #padS = np.zeros((B_x, B_y))
    #padS[:L_xi, :L_eta] = wave_field_start
    #G = fftshift(fft2(padS))
    G = fftshift(fft2(wave_field_start))

    # frequency sampling
    f_x = np.arange(-B_x + 1, B_x + 1, delta_x)  # fixme +1 necessary?
    f_y = np.arange(-B_y + 1, B_y + 1, delta_y)
    f_x, f_y = np.meshgrid(f_x, f_y)

    H = np.exp(1j * k * z - np.pi * 1j * wavelength * z * (f_x ** 2 + f_y ** 2))  # fourier transformed Fresnel convolution kernel.
    #h = - 1 / (1j * wavelength * z) * np.exp(1j * k * z + 1j * np.pi / (z * wavelength) * (f_x ** 2 + f_y ** 2))  # Fresnel convolution kernel
    # H = fftshift(fft2(h))

    tmp = ifft2(ifftshift(G * H))
    return tmp[:L_x, :L_y]
    