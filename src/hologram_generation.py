import numpy as np
from numpy import pi
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import time

from src.propagation_models import fresnel_propagation


def time_averaged_gs_3d_hologram_generation(target_volume, plane_locations, incident_field, wavelength, hologram_stack_size=100, gs_iterations=3):

    timing_start = time.time()
    print("{} Hologram generation start".format(time.strftime("%H:%M:%S", time.localtime(timing_start))))
    hologram_pattern_stack = np.zeros(target_volume.shape[:2] + (hologram_stack_size,), dtype=np.complex_)

    for hologram_index in range(hologram_stack_size):
        hologram_pattern = np.ones(target_volume.shape[:2]) * np.exp(2j * pi * np.random.random(target_volume.shape[:2]))  # init holo with amplitude of 1 and random phase

        # Gercherg-Saxton algorithm
        # todo: determine if averaging planes into one works
        for gs_it in range(gs_iterations):
            wave_field_DMD = np.zeros(target_volume.shape, dtype=complex)

            for plane_index in range(target_volume.shape[2]):  # for each plane in target volume
                DMD_constraint = incident_field * np.exp(1j * np.angle(hologram_pattern))  #  Phase modulation. unknown possible with DMD? try amplitude (transmittance) modulation
                wave_field_object = fresnel_propagation(fftshift(fft2(DMD_constraint)), wavelength, plane_locations[plane_index])  # Fourier-lens propagation followed by near-field Fresnel propagation
                target_constraint = target_volume[:, :, plane_index] * np.exp(1j * np.angle(wave_field_object))  # replace intensity by target one. Keep phase.
                wave_field_DMD[:, :, plane_index] = ifft2(ifftshift(fresnel_propagation(target_constraint, wavelength, -plane_locations[plane_index])))  # "inverse" fresnel propagation

            # Phase is mean of phases from each planes. unknown: how can one average produce correct 3D target??
            hologram_pattern = np.ones(target_volume.shape[:2]) * np.exp(1j * np.mean(np.angle(wave_field_DMD), axis=2))  # hologram has amplitude 1 bc we don't care about amplitude here (phase modulation)

            print("\tGS Iteration {}/{}".format(gs_it + 1, gs_iterations))

        hologram_pattern_stack[:, :, hologram_index] = hologram_pattern

        print("Hologram {}/{}".format(hologram_index + 1, hologram_stack_size))

    timing_end = time.time()
    print("{} Hologram generation done in {}".format(time.strftime("%H:%M:%S", time.localtime(timing_end)), time.strftime("%Mmin %Ss", time.localtime(timing_end - timing_start))))

    return hologram_pattern_stack



