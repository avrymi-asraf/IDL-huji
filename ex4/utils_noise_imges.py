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
    x_true = x[:32].reshape(32,1,32,32).copy()
    kernel = np.array([0.25, 0.5, 0.25])
    for i in range(32):
        x_true[i, 0, :, :] = np.convolve(x_true[i, 0, :, :], kernel, mode='same')
    x_true = x_true.swapaxes(1, 2)
    for i in range(32):
        x_true[i, 0, :, :] = np.convolve(x_true[i, 0, :, :], kernel, mode='same')
    x_true = x_true.swapaxes(1, 2)
    if twice:
        return blur_image(x_true, False)
    return x_true