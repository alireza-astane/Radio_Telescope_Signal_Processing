# Signal Processing for Radio Telescope Project

This project, completed under the supervision of Dr. Reza Rezaee at Sharif University of Technology, simulates a real-time signal processing pipeline for multiple radio telescopes. The system:
- Accepts real-time data streams from telescopes.
- Calculates phase differences between signals.
- Tunes and overlaps the signals.
- Computes the Fourier transform to generate a sky image.
- Identifies optimal positions for multiple radio telescopes.

## Repository Structure

- **C Sources:**
  - [process.c](process.c): Reads signal data, performs Hilbert transform ([`hilbertTransform`](process.c)).
  - [client4.c](client4.c) & [client5.c](client5.c): Implement client routines for receiving data, performing FFT using AVX instructions (see [fft_avx.c](fft_avx.c) and [fft_basic.c](fft_basic.c)) and computing phase differences.
  - [argmax.c](argmax.c) & [cumsum.c](cumsum.c): Utility functions.
  - [saver.c](saver.c), [server2.c](server2.c), [server2b.c](server2b.c): Additional signal processing and networking routines.
  
- **Notebooks:**
  - [hilbert.ipynb](hilbert.ipynb) & [methods.ipynb](methods.ipynb): Demonstrate signal processing methods (Hilbert transform, FFT, cross-correlation). These notebooks also illustrate the use of **Torch** for GPU-accelerated tensor computations and **Numba** to optimize numerical Python code.

- **Build System:**
  - The [build/](build/) directory contains CMake configuration files.
  - The provided Makefile (see below) compiles the core C executables.

- **Libraries & Figures:**
  - **libtfhe/** and **tfhe/**: Contain supporting libraries.
  - **figs/**: Contains generated plots and images.

## Tools & Technologies

- **AVX:**  
  Many computationally intensive routines (like in [fft_avx.c](fft_avx.c) and [client4.c](client4.c)) use AVX instructions for SIMD parallelism to accelerate vectorized math operations.

- **Torch:**  
  Used in Python notebooks ([methods.ipynb](methods.ipynb)) for fast, GPU-accelerated mathematical computing and simulations (e.g., computing sine functions on large tensors).

- **Numba:**  
  Applied to speed up Python numerical functions with just-in-time compilation, reducing runtime in signal processing algorithms.

## How to Use This Repository

1. **Building the C Executables:**

   Open a terminal in the project root and run:
   
   ```sh
   make