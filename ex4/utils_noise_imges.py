import numpy as np

"""
take a MNSIT image and black out a half or a quarter of the pixels, depending on the value of the noise parameter.
"""
def noise_image_quaters(x, half: bool):
    x_true = x[:32].reshape(32,1,32,32).copy()
    if half:
        x_true[:, :, 16:, :] = 0
    else:
        x_true[:, :, 16:, 16:] = 0
    return x_true

"""
Blur a MNIST image by applying the 2D filter resulting from this convolution of 1D filters [0.25, 0.5, 0.25]^T*[0.25, 0.5, 0.25]
once or twice, depending on the value of the noise parameter. 
"""
def blur_image(x, twice: bool):
    x_true = x[:32].reshape(32,32,32).copy()
    kernel = np.array([0.25, 0.5, 0.25])
    kernel = np.outer(kernel, kernel)
    x_true = cv2.filter2D(x_true, -1, kernel)
    if twice:
        x_true = cv2.filter2D(x_true, -1, kernel)
    x_true = x_true[:32].reshape(32, 1, 32, 32)
    return x_true



def estimate_vanilla_noise_level(degraded_images, clean_images):
    # Placeholder function for estimating noise level
    noise_levels = []
    for degraded, clean in zip(degraded_images, clean_images):
        noise = degraded - clean
        noise_std = np.std(noise)
        signal_std = np.std(clean)
        noise_level = noise_std / np.sqrt(noise_std**2 + signal_std**2)
        noise_levels.append(noise_level)
    return np.mean(noise_levels)


import cv2
from skimage.restoration import denoise_tv_chambolle


def estimate_noise_tv(degraded_images, clean_images):
    noise_levels = []
    for degraded, clean in zip(degraded_images, clean_images):
        tv_clean = denoise_tv_chambolle(clean, weight=0.1)
        tv_degraded = denoise_tv_chambolle(degraded, weight=0.1)

        noise_tv = np.sum(np.abs(tv_degraded - tv_clean))
        signal_tv = np.sum(np.abs(tv_clean))

        noise_level = noise_tv / signal_tv
        noise_levels.append(noise_level)

    return 1 - np.mean(noise_levels)

from scipy.fft import fft2, ifft2, fftshift
def estimate_noise_nps(degraded_images, clean_images):
    def compute_power_spectrum(image):
        # Compute the 2D Fourier Transform of the image
        f_transform = fft2(image)
        # Shift the zero-frequency component to the center
        f_transform_shifted = fftshift(f_transform)
        # Compute the power spectrum
        power_spectrum = np.abs(f_transform_shifted) ** 2
        return power_spectrum

    noise_levels = []
    for degraded, clean in zip(degraded_images, clean_images):
        residual = degraded - clean

        # Compute the power spectrum of the residual
        residual_power_spectrum = compute_power_spectrum(residual.squeeze())

        # Estimate noise level from the power spectrum
        noise_level = np.mean(residual_power_spectrum)
        noise_levels.append(noise_level)

    return np.mean(noise_levels) * 5e-3 # todo: might need to adjust this factor