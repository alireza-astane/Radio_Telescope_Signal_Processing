# Legacy prototypes (2024)

These are the original files from the bachelor project, kept unchanged so the
evolution of the work is visible. They are **not** built or tested; the
maintained implementation lives in [`src/radiotel`](../src/radiotel) and
[`csrc`](../csrc).

| File | What it was | Known issues (fixed in the new code) |
|------|-------------|---------------------------------------|
| `server2.c`, `server2b.c` | Two TCP "receivers" streaming sine waves on ports 12345/12346 (the second with a +1 rad phase offset). | Single client, no framing/sequence numbers, sends identical buffers every time. |
| `client4.c` | Connects to both servers, Hilbert transform via FFTW, unwraps phase, estimates mean phase difference, rolls & multiplies the signals. Also contains AVX experiments (prefix sum, unwrap, dot product, roll). | Receiver threads lock the mutex only *after* `recv` has already written the buffer (data race); FFTW plans rebuilt every block; `unwrap_phase3` AVX path unfinished (scalar fallback overwrites it); `roll`-by-phase only valid for a single tone. |
| `client5.c` | Same pipeline using the phase of the FFT peak bin instead of Hilbert. | Same threading issues; phase of a single bin is biased for off-bin tones. |
| `fft_basic.c`, `fft_avx.c` | Recursive radix-2 FFT and an AVX butterfly attempt. | `M_PI` redefined as 3.1415; the AVX version does no bit-reversal and wrongly broadcasts one twiddle per 4 butterflies, so it is not a correct FFT. |
| `cumsum.c`, `argmax.c` | AVX prefix-sum and argmax experiments. | `cumsum.c` mixes `__m256i`/`__m256d` and does not compile cleanly. |
| `process.c`, `saver.c` | O(N²) direct-convolution Hilbert transform on files written by `saver.c`. | Quadratic; stack arrays of 10⁴ doubles ×4. |
| `methods.ipynb` | Comparison of FFT / Hilbert / cross-correlation phase estimators (error vs frequency, noise, data rate, duration), Torch/CUDA timing, **u-v tracks from Earth rotation (`getV`)**, baseline construction (`get_Bs`) and a random-search / annealing array-layout optimizer. | Cross-correlation error quantised to whole samples; errors not phase-wrapped; layout score computed by counting matplotlib pixels; `get_Bs` hard-coded for 4/6 antennas. |
| `hilbert.ipynb` | From-scratch FFT/IFFT/Hilbert in NumPy, checked against SciPy. | — |
| `Makefile`, `command.txt` | Original build notes. | Makefile links ten files that each define `main()`. |
| `figs/` | Figures produced by `methods.ipynb` (u-v tracks for line/square layouts, 6-telescope layout). | — |

The ideas that survived — pairwise baselines, Earth-rotation synthesis,
layout optimisation and fast pairwise phase computation — are generalised
in the new package; see the top-level README for the mapping.
