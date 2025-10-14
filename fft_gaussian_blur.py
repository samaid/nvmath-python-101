import numpy as np
import cupy as cp
import nvmath
from PIL import Image


sigma_value = 20


def create_gaussian_kernel_2d_r2c(shape, sigma):
    """
    Create the Gaussian kernel's frequency response for R2C FFT.
    For R2C FFT, we only need the positive frequencies in the last dimension.
    """
    h, w = shape

    # frequency coordinates in cycles/sample for each axis (CuPy)
    fy = cp.fft.fftfreq(h)[:, None]  # column vector
    fx = cp.fft.rfftfreq(w)[None, :]  # row vector for R2C (only positive frequencies)

    # Continuous Fourier transform of a Gaussian g(x)=exp(-x^2/(2*sigma^2)) is another Gaussian
    # G(f) = exp(-2 * pi^2 * sigma^2 * f^2). For 2D separable:
    # H(fx,fy) = exp(-2 * pi^2 * sigma^2 * (fx^2 + fy^2)).
    H = cp.exp(-2.0 * cp.pi * cp.pi * sigma * sigma * (fx * fx + fy * fy))
    return H


def process_batch_nvmath(images_gpu, sigma):
    """
    Process a batch of images using nvmath with stateful API.

    This function uses the stateful API to create a persistent FFT plan
    and epilog that can be reused across multiple images in the batch.
    """

    # Create the gaussian kernel once for the batch (R2C format)
    kernel_fft = create_gaussian_kernel_2d_r2c(images_gpu[0].shape, sigma).astype(cp.complex64)
    wh = images_gpu[0].shape[0] * images_gpu[0].shape[1]

    # Define epilog function for gaussian kernel multiplication
    def gaussian_multiply(data_out, offset, data, kernel_data, unused):
        """Epilog function to multiply FFT data with gaussian kernel."""
        data_out[offset] = data * kernel_data[offset] / wh  # Normalize by the image area

    # Compile the epilog to LTO-IR once
    with cp.cuda.Device():
        epilog = nvmath.fft.compile_epilog(gaussian_multiply, "complex64", "complex64")

    def convolve_gpu(fft, ifft, image_gpu):
        fft.reset_operand(image_gpu)
        image_fft = fft.execute(direction=nvmath.fft.FFTDirection.FORWARD)
        ifft.reset_operand(image_fft)
        image_ifft = ifft.execute(direction=nvmath.fft.FFTDirection.INVERSE)
        return image_ifft

    image_gpu = images_gpu[0]  # Real input for R2C FFT
    c2r_output = cp.empty((image_gpu.shape[0], image_gpu.shape[1] // 2 + 1), dtype=cp.complex64)

    with (
        nvmath.fft.FFT(image_gpu, options={"fft_type": "R2C"}) as fft,
        nvmath.fft.FFT(c2r_output, options={"fft_type": "C2R"}) as ifft,
    ):

        # Two plans are created, one for the forward R2C FFT with an epilog and another for the inverse C2R FFT
        fft.plan(epilog={"ltoir": epilog, "data": kernel_fft.data.ptr})

        ifft.plan()

        # Process each image in the batch
        filtered_images = []
        for i in range(len(images_gpu)):
            filtered_images.append(convolve_gpu(fft, ifft, images_gpu[i]))
    return filtered_images


def gaussian_filter_cupy(image, sigma, clear_cache=True):
    """
    Apply Gaussian filter using CuPy R2C/C2R FFT.
    """
    if clear_cache:
        cp.fft.config.clear_plan_cache()  # Clear CuPy FFT cache to ensure clean FFT benchmarking
    image = cp.asarray(image, dtype=cp.float32)
    kernel_fft = create_gaussian_kernel_2d_r2c(image.shape, sigma)
    image_fft = cp.fft.rfft2(image)  # Real to complex FFT
    filtered = cp.fft.irfft2(image_fft * kernel_fft, s=image.shape)  # Complex to real FFT
    return cp.asnumpy(filtered)


def process_batch_cupy(images_gpu, sigma_value):
    """
    Process a batch of images using CuPy.
    """
    filtered_images = []
    for i in range(len(images_gpu)):
        filtered_images.append(gaussian_filter_cupy(images_gpu[i], sigma_value, clear_cache=False))
    return filtered_images


from cupyx.profiler import benchmark

asset_path = "./"
img = Image.open(asset_path + "dog.jpg").convert("L")
original_image = np.array(img, dtype=np.float32) / 255.0

batch_size = 1 
images_gpu = [cp.asarray(original_image, dtype=cp.float32)] * batch_size

# Process the batch using nvmath stateful API
print(benchmark(lambda: process_batch_cupy(images_gpu, sigma_value), n_repeat=5, n_warmup=1))
print(benchmark(lambda: process_batch_nvmath(images_gpu, sigma_value), n_repeat=5, n_warmup=1))

