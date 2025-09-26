from numba import cuda
from nvmath.device import random
import cupy as cp
from cupyx.profiler import benchmark
import math

RNG_SEED = 777777 # Random seed
N_STEPS = 252 # Number of time steps (trading days in a year)
N_PATHS = 800000 # Number of simulated paths (large number to get a reliable estimate)

S0 = 100.0 # Initial stock price
MU = 0.003 # Drift with upward trend
SIGMA = 0.027 # Volatility

def brownian_motion(nsteps, npaths, mu, sigma):
    # Differential form of the Brownian motion
    dBt = cp.empty((npaths, nsteps), dtype=cp.float32, order='F')
    dBt[:, 0] = 0.0 # The process starts at 0
    dBt[:, 1:] = cp.random.randn(npaths, nsteps - 1) * sigma + mu

    # Integral form of the Brownian motion
    Bt = cp.cumsum(dBt, axis=1)

    return Bt

def generate_gbm_paths_cupy(npaths, nsteps, mu, sigma, s0):
    b_t = brownian_motion(nsteps, npaths, mu, sigma)
    paths = s0 * cp.exp(b_t)
    return paths

cp.random.seed(RNG_SEED)
s_t = generate_gbm_paths_cupy(N_PATHS, N_STEPS, MU, SIGMA, S0)

print("CUPY ==========================================================")
print(f"Mean stock price at t=T: {s_t[:, -1].mean():0.2f}")
print(f"Standard deviation of stock price at t=T: {s_t[:, -1].std():0.2f}")
print(benchmark(lambda: generate_gbm_paths_cupy(N_PATHS, N_STEPS, MU, SIGMA, S0), n_repeat=5, n_warmup=1))

# Pre-compile the random number generator into IR to use alongside other device code
compiled_rng = random.Compile(cc=None)

# Set up CUDA kernel launch configuration
paths_per_thread = 1 # Each thread generates paths_per_thread paths
threads_per_block = 32 # Number of threads per block
blocks = N_PATHS // (threads_per_block * paths_per_thread) # Number of blocks
nthreads = threads_per_block * blocks # Total number of threads
print(f"blocks: {blocks}, threads_per_block: {threads_per_block}, nthreads: {nthreads}")

# Allocate space for random states
states = random.StatesPhilox4_32_10(N_PATHS)

# RNG initialization kernel
@cuda.jit(link=compiled_rng.files, extensions=compiled_rng.extension)
def init_rng_gpu(states, seed):
    idx = cuda.grid(1)
    path_idx = idx * paths_per_thread
    for k in range(path_idx, min(path_idx + paths_per_thread, N_PATHS)):
        random.init(seed, k, 0, states[k])

@cuda.jit(link=compiled_rng.files, extensions=compiled_rng.extension)
def generate_gbm_paths_nvmath(states, paths, nsteps, mu, sigma, s0):
    mu = paths.dtype.type(mu)
    sigma = paths.dtype.type(sigma)
    s0 = paths.dtype.type(s0)
    
    idx = cuda.grid(1)
    path_idx = idx * paths_per_thread

    for k in range(path_idx, min(path_idx + paths_per_thread, N_PATHS)):
        paths[k, 0] = s0

        # Consume 4 normal variates at a time for better throughput
        for i in range(1, nsteps, 4):
            v = random.normal4(states[k])  # Returned as float32x4 type
            vals = v.x, v.y, v.z, v.w  # Decompose into a tuple of float32
            # Process a chunk of 4 time steps, use min() to avoid out-of-bounds access
            for j in range(i, min(i + 4, nsteps)):
                paths[k, j] = paths[k, j - 1] * math.exp(mu + sigma * vals[j - i])


# Allocate space for paths
paths_gpu = cp.empty((N_PATHS, N_STEPS), dtype=cp.float32, order='F')

# Initialize RNG states
init_rng_gpu[blocks, threads_per_block](states, RNG_SEED)

# Generate GBM paths on GPU
generate_gbm_paths_nvmath[blocks, threads_per_block](
    states, paths_gpu, N_STEPS, MU, SIGMA, S0
)

print("NVMATH-PYTHON ===============================================================")
print(f"Mean stock price at t=T: {paths_gpu[:, -1].mean():0.2f}")
print(f"Standard deviation of stock price at t=T: {paths_gpu[:, -1].std():0.2f}")
print(benchmark(lambda: generate_gbm_paths_nvmath[blocks, threads_per_block](states, paths_gpu, N_STEPS, MU, SIGMA, S0), n_repeat=5, n_warmup=1))

