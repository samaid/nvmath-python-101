from PIL import Image
from scipy.ndimage import gaussian_filter
import numpy as np
import nvmath
import cupy as cp

asset_path = "./"
img = Image.open(asset_path + "dog.jpg").convert("L")
original_image = np.array(img, dtype=np.float32) / 255.0

sigma_value = 20.0  # Filter size

batch_size = 1 
images_gpu = [cp.asarray(original_image, dtype=cp.float32)] * batch_size

def create_gaussian_kernel_2d_r2c(shape, sigma):
    """
    Create the Gaussian kernel's frequency response for R2C FFT.
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
    H = cp.exp(-2.0 * cp.pi * cp.pi * sigma * sigma * (fx * fx + fy * fy))
    return H# A simple class to allow caching and resource management


class FFTCache(dict):
    def free(self):
        """Release all resources owned in the cache"""
        for fft_obj in self.values():
            fft_obj.free()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.free()


def process_batch_nvmath(images_gpu, sigma):
    """
    Process a batch of images using nvmath with stateful API.

    This function uses the stateful API to create a persistent FFT plan
    and epilog that can be reused across multiple images in the batch.
    """


    # A simple illustration of creating and using a cached FFT operation.
    def cached_fft(
        a,
        axes=None,
        direction=None,
        options=None,
        execution=None,
        prolog=None,
        epilog=None,
        stream=None,
        cache: dict | None = None,
    ):
        """
        A cached version of FFT, taking a cache argument in addition to the regular arguments
        for fft(). The stateful objects are cached in the provided cache, and reused.

        Args:
            cache: an object to use as the cache that satisfies `typing.Mapping` concept.

        Note:
            User is responsible for explicitly free all resources stored in `cache` after no
            longer needed. If a native `dict` object is used to store the cache, the resources
            can be released via:

            >>> for f in cache.values():
            >>>    f.free()

            Alternatively, users may use the `FFTCache` class above. Resources can be cleaned by
            a call the the `free` method or will be automatically released if used in a context
            manager.
        """
        if cache is None:
            cache = {}
        # logger = logging.getLogger()

        package = stream.__class__.__module__.split(".")[0]
        stream_ptr = stream.ptr if package == "cupy" else stream.cuda_stream if package == "torch" else stream

        key = nvmath.fft.FFT.create_key(a, axes=axes, options=options, execution=execution, prolog=prolog, epilog=epilog)

        # Get object from cache if it already exists, or create a new one and add it to the
        # cache.
        if (key, stream_ptr) in cache:
            # logger.info("Cache HIT: using planned object.")
            # The planned object is already cached, so retrieve it.
            f = cache[key, stream_ptr]
            # Set new operand in object.
            f.reset_operand(a, stream=stream)
        else:
            # Create a new stateful object, plan the operation, and cache the  object.
            f = cache[key, stream_ptr] = nvmath.fft.FFT(a, axes=axes, options=options, execution=execution, stream=stream)
            f.plan(prolog=prolog, epilog=epilog, stream=stream)
            # logger.info("Cache MISS: creating and caching a planned FFT object.")

        # Execute the FFT on the cached object.
        r = f.execute(direction=direction, stream=stream)

        # Reset operand to None to discard internal reference, allowing memory to be recycled.
        f.reset_operand(stream=stream)

        return r


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

    def convolve_gpu(image_gpu):
        image_fft = cached_fft(image_gpu, direction=nvmath.fft.FFTDirection.FORWARD, epilog={"ltoir": epilog, "data": kernel_fft.data.ptr})
        image_fft = cached_fft(image_fft, options={"fft_type": "C2R"})
        return image_fft

    image_gpu = images_gpu[0]  # Real input for R2C FFT
    c2r_output = cp.empty((image_gpu.shape[0], image_gpu.shape[1] // 2 + 1), dtype=cp.complex64)

    # with (
    #     nvmath.fft.FFT(image_gpu, options={"fft_type": "R2C"}) as fft,
    #     nvmath.fft.FFT(c2r_output, options={"fft_type": "C2R"}) as ifft,
    # ):
    #     # Two plans are created, one for the forward R2C FFT with an epilog
    #     # and another for the inverse C2R FFT
    #     fft.plan(epilog={"ltoir": epilog, "data": kernel_fft.data.ptr})
    #     ifft.plan()

    # Process each image in the batch
    filtered_images = []
    for i in range(len(images_gpu)):
        filtered_images.append(convolve_gpu(images_gpu[i]))
    return filtered_images


# Process the batch using nvmath stateful API
filtered_images_nvmath = process_batch_nvmath(images_gpu, sigma_value)

from cupyx.profiler import benchmark

print(benchmark(lambda: process_batch_nvmath(images_gpu, sigma_value), n_repeat=5, n_warmup=1))

