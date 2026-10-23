# Benchmark methodology

All numbers come from scripts in `benchmarks/`. Results are committed under
`benchmarks/results/`, and each correlator row records the CPU model and git
commit. Figures and the tables in [benchmark_tables.md](benchmark_tables.md)
are regenerated from those CSVs with `make figures`.

## Correlator (`bench_correlator.py`)

* **Input:** independent complex Gaussian noise, `complex64`, with
  2¹⁸ = 262,144 samples per antenna (0.11 s of data at 2.4 MS/s).
* **Sweep:** N ∈ {2, 4, 8, 16, 32, 64} antennas × nchan ∈ {256, 1024, 4096}
  channels × every available backend.
* **Timing:** best of 3 runs after one warm-up. The F-step (channelisation
  FFT) and X-step are timed separately on the same spectra. Torch timings
  call `torch.cuda.synchronize()` on GPU, and host↔device transfers are not
  included in the per-step times.
* **Metrics:**
  * `gflops_x` = 8 · n_bl · n_samples / t_x. Each complex multiply-accumulate
    is 8 real flops.
  * `rtf_xstep` = (n_samples / 2.4 MS/s) / t_x is the real-time factor of
    the X-step alone. Values > 1 mean that stage keeps up with a live
    stream.
  * `rtf_total` includes the F-step.
* **Skipped configurations:** the slow references (`numpy-loop`,
  `c-scalar`) are skipped when their estimated time exceeds 20 s. They
  appear as "—" in the tables.
* **Thread scaling:** `c-avx2-omp` at N = 32, 1024 channels, with 1…all
  threads.
* **GPU:** the CPU host had no working CUDA driver, so the `torch-cuda`
  rows were measured separately on a Colab GPU with
  [`benchmarks/gpu_colab.ipynb`](../benchmarks/gpu_colab.ipynb)
  (`--gpu-only`). They are stored in `correlator_gpu.csv`. Those rows reflect
  a different machine, and newer runs record the GPU model in the `device`
  column. `make figures` merges those rows into the plots and tables.
  Without that file, `torch-cuda` appears as `status = pending`.

## Phase estimators (`bench_phase.py`)

* **Signals:** two real tones at 97.3 Hz (deliberately off-bin) sampled at
  1024 Hz, with a true phase difference of 1 rad. Independent white noise is
  added to each, with SNR defined as A²/2σ² per channel.
* **Accuracy vs SNR:** −15…30 dB, 400 Monte-Carlo trials per point, RMS of
  the wrapped error. Compared with the Cramér–Rao bound √(2/(N·SNR)).
* **Accuracy vs frequency:** a noise-free (30 dB) sweep from near DC to
  Nyquist, showing edge effects.
* **Runtime:** best of 5 runs for N = 2⁸…2²⁰ samples.

## Streaming (`bench_stream.sh`)

* **Setup:** `rx_sim` runs one thread and one TCP connection per receiver on
  localhost and sends 4096-sample frames of 8-bit I/Q. `corrd` correlates
  them with 64 channels and 16-frame integrations, using the AVX2+OpenMP
  kernel.
* **Unlimited mode:** senders go as fast as TCP back-pressure allows. The
  sustained MS/s per receiver is the daemon's maximum throughput, including
  socket I/O, unpacking, the FFT and the X-step. The run uses N ∈ {2, 4, 8,
  16, 32} and 2000 frames each.
* **Paced mode:** senders run at the nominal 2.4 MS/s per receiver for
  1200 frames (about 2 s). A non-zero *late* count would mean the
  correlator falls behind real time.
* **Correctness:** a simulated observation is quantised to 8 bits, streamed
  through the daemon, and its dumps are compared with
  `correlate(..., "numpy-gemm")` on the same samples. The script fails if
  the relative difference exceeds 1e-4.

Caveats: everything runs on one machine, so the senders compete with the
correlator for CPU and the loopback "network" is far faster than a campus
link. Distributed deployment, NIC throughput and clock synchronisation
between real receivers are out of scope. See the report's *Limitations*
section.
