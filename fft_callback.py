import nvmath
import numpy as np
import cupy as cp
from PIL import Image
from cupyx.profiler import benchmark

img = Image.open("./dog.jpg").convert('L')
original_image = np.array(img, dtype=np.float32) / 255.0

sigma_value = 20.0 # Filter size


def create_gaussian_filter(shape, sigma):
    """
    Create the Gaussian filter's frequency response for R2C FFT.
    For R2C FFT, we only need the positive frequencies in the last dimension.
    """
    h, w = shape

    # frequency coordinates in cycles/sample for each axis (CuPy)
    fy = cp.fft.fftfreq(h)[:, None]  # column vector
    fx = cp.fft.rfftfreq(w)[None, :]  # row vector for R2C (only positive frequencies)

    # Continuous Fourier transform of a Gaussian
    # g(x)=exp(-x^2/(2*sigma^2)) is another Gaussian
    # G(f) = exp(-2 * pi^2 * sigma^2 * f^2). For 2D separable:
    # H(fx,fy) = exp(-2 * pi^2 * sigma^2 * (fx^2 + fy^2)).
    h = cp.exp(-2.0 * cp.pi * cp.pi * sigma * sigma * (fx * fx + fy * fy))
    return h


def gaussian_filter_cupy(image, sigma, clear_cache=True):
    """
    Apply Gaussian filter using CuPy R2C/C2R FFT.
    """
    if clear_cache:
        cp.fft.config.clear_plan_cache()  # Clear CuPy FFT cache to ensure clean FFT benchmarking
    filter = create_gaussian_filter(image.shape, sigma)
    image_fft = cp.fft.rfft2(image)  # Real to complex FFT
    filtered = cp.fft.irfft2(image_fft * filter, s=image.shape)  # Complex to real FFT
    return filtered


image_gpu = cp.asarray(original_image, dtype=cp.float32)
filtered_image_cupy = gaussian_filter_cupy(image_gpu, sigma_value)
print("***CUPY***")
print(f"Checksum={filtered_image_cupy.sum():.1f}")
print(benchmark(lambda: gaussian_filter_cupy(image_gpu, sigma_value), n_repeat=5, n_warmup=1))


def gaussian_filter_nvmath(image, sigma):
    """
    Apply Gaussian filter using nvmath FFT helper functions on GPU with epilog.

    This function uses nvmath.fft.rfft / nvmath.fft.irfft with an epilog to perform
    the gaussian kernel multiplication directly within the FFT operation.
    The input is moved to GPU as a CuPy real array and results are returned as NumPy array.
    """
    wh = image.shape[0] * image.shape[1]

    # Gaussian kernel on GPU for R2C FFT
    filter = create_gaussian_filter(image.shape, sigma)

    # Define epilog function for gaussian kernel multiplication
    def epilog_impl(data_out, offset, data, filter_data, unused):
        """Epilog function to multiply FFT data with gaussian kernel."""
        data_out[offset] = data * filter_data[offset] / wh  # Normalize by the image area

    # Compile the epilog to LTO-IR
    epilog = nvmath.fft.compile_epilog(epilog_impl, "complex64", "complex64")

    # Compute R2C FFT using nvmath with epilog to apply gaussian kernel multiplication
    image_fft = nvmath.fft.rfft(image, epilog={"ltoir": epilog, "data": filter.data.ptr})

    # Inverse C2R FFT using nvmath
    filtered = nvmath.fft.irfft(image_fft)

    return filtered

filtered_image_nvmath = gaussian_filter_nvmath(image_gpu, sigma_value)
print("*** NVMATH ***")
print(f"Checksum={filtered_image_cupy.sum():.1f}")
print(benchmark(lambda: gaussian_filter_nvmath(image_gpu, sigma_value), n_repeat=5, n_warmup=1))


batch_size = 16
images_gpu = [image_gpu] * batch_size


def process_batch_cupy(images_gpu, sigma_value):
    """
    Process a batch of images using CuPy.
    """
    # Clear CuPy FFT cache to ensure clean FFT benchmarking during multiple repetitions (n_repeat > 1)
    # For fair comparison, we do not want the planning cost of the first call to be ignored
    cp.fft.config.clear_plan_cache()
    filtered_images = []
    for i in range(len(images_gpu)):
        filtered_images.append(gaussian_filter_cupy(images_gpu[i], sigma_value, clear_cache=False))
    return filtered_images


filtered_images_cupy = process_batch_cupy(images_gpu, sigma_value)
print("*** CUPY BATCHED ***")
print(f"Checksum={filtered_images_cupy[0].sum():.1f}")
print(benchmark(lambda: process_batch_cupy(images_gpu, sigma_value), n_repeat=5, n_warmup=1))


def process_batch_nvmath(images_gpu, sigma):
    """
    Process a batch of images using nvmath with stateful API.

    This function uses the stateful API to create a persistent FFT plan
    and epilog that can be reused across multiple images in the batch.
    """

    # Create the gaussian kernel once for the batch (R2C format)
    filter = create_gaussian_filter(images_gpu[0].shape, sigma).astype(cp.complex64)
    wh = images_gpu[0].shape[0] * images_gpu[0].shape[1]


    # Define epilog function for gaussian kernel multiplication
    def epilog_impl(data_out, offset, data, filter_data, unused):
        """Epilog implementation."""
        data_out[offset] = data * filter_data[offset] / wh  # Normalize by the image area


    # Compile the epilog to LTO-IR once
    epilog = nvmath.fft.compile_epilog(epilog_impl, "complex64", "complex64")


    def convolve_gpu(fft, ifft, image_gpu):
        fft.reset_operand(image_gpu)
        image_fft = fft.execute(direction=nvmath.fft.FFTDirection.FORWARD)
        ifft.reset_operand(image_fft)
        image_ifft = ifft.execute(direction=nvmath.fft.FFTDirection.INVERSE)
        return image_ifft


    image_gpu = images_gpu[0]  # Real input for R2C FFT
    image_fft = cp.empty((image_gpu.shape[0], image_gpu.shape[1] // 2 + 1), dtype=cp.complex64)

    with (
        nvmath.fft.FFT(image_gpu) as fft,
        nvmath.fft.FFT(image_fft, options={"fft_type": "C2R"}) as ifft,
    ):
        # Two plans are created, one for the forward R2C FFT with an epilog
        # and another for the inverse C2R FFT
        fft.plan(epilog={"ltoir": epilog, "data": filter.data.ptr})
        ifft.plan()

        # Process each image in the batch
        filtered_images = []
        for i in range(len(images_gpu)):
            filtered_images.append(convolve_gpu(fft, ifft, images_gpu[i]))
    return filtered_images


# Process the batch using nvmath stateful API
filtered_images_nvmath = process_batch_nvmath(images_gpu, sigma_value)
print("*** NVMATH BATCHED ***")
print(f"Checksum={filtered_images_nvmath[0].sum():.1f}")
print(benchmark(lambda: process_batch_nvmath(images_gpu, sigma_value), n_repeat=5, n_warmup=1))


def process_cupy_first_call(image_gpu, sigma_value):
    # To emulate the cost of the first call we need to clear the cache
    cp.fft.config.clear_plan_cache()
    gaussian_filter_cupy(image_gpu, sigma_value, clear_cache=False)

def process_cupy_subsequent_call(image_gpu, sigma_value):
    # With n_repeat > 1 the first call cost will be ignored
    gaussian_filter_cupy(image_gpu, sigma_value, clear_cache=False)


print("*** CUPY BREAKDOWN ***")
print("First call", benchmark(lambda: process_cupy_first_call(image_gpu, sigma_value), n_repeat=5, n_warmup=1))
print("Subsequent calls", benchmark(lambda: process_cupy_subsequent_call(image_gpu, sigma_value), n_repeat=5, n_warmup=1))


# Create the gaussian kernel once for the batch (R2C format)
filter = create_gaussian_filter(image_gpu.shape, sigma_value).astype(cp.complex64)
wh = image_gpu.shape[0] * image_gpu.shape[1]

# Define epilog function for gaussian kernel multiplication
def epilog_impl(data_out, offset, data, filter_data, unused):
    """Epilog implementation."""
    data_out[offset] = data * filter_data[offset] / wh  # Normalize by the image area

# Compile the epilog to LTO-IR once
def compile_epilog():
    return nvmath.fft.compile_epilog(epilog_impl, "complex64", "complex64")

def forward_fft_execute(fft, image_gpu):
    fft.reset_operand(image_gpu)
    return fft.execute(direction=nvmath.fft.FFTDirection.FORWARD)

def inverse_fft_execute(ifft, image_gpu):
    ifft.reset_operand(image_gpu)
    return ifft.execute(direction=nvmath.fft.FFTDirection.INVERSE)

def forward_fft_plan(image_gpu, epilog):
    fft = nvmath.fft.FFT(image_gpu)
    fft.plan(epilog={"ltoir": epilog, "data": filter.data.ptr})
    return fft

def inverse_fft_plan(c2r_output):
    ifft = nvmath.fft.FFT(c2r_output, options={"fft_type": "C2R"})
    ifft.plan()
    return ifft

epilog = compile_epilog()
c2r_output = cp.empty((image_gpu.shape[0], image_gpu.shape[1] // 2 + 1), dtype=cp.complex64)
fft = forward_fft_plan(image_gpu, epilog)
ifft = inverse_fft_plan(c2r_output)
fft_image = forward_fft_execute(fft, image_gpu)
filtered_image = inverse_fft_execute(ifft, fft_image)

print(image_gpu.sum())
print(filtered_image.sum())

print(f"Compilation cost = ", benchmark(lambda: compile_epilog(), n_repeat=5, n_warmup=1))
print(f"Forward FFT plan cost =", benchmark(lambda: forward_fft_plan(image_gpu, epilog), n_repeat=5, n_warmup=1))
print(f"Inverse FFT plan cost =", benchmark(lambda: inverse_fft_plan(c2r_output), n_repeat=5, n_warmup=1))
print(f"Forward FFT execute cost =", benchmark(lambda: forward_fft_execute(fft, image_gpu), n_repeat=5, n_warmup=1))
print(f"Inverse FFT execute cost =", benchmark(lambda: inverse_fft_execute(ifft, fft_image), n_repeat=5, n_warmup=1))
