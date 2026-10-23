# Architecture

```
                    ┌──────────────────────── offline / simulation (Python) ────────────────────────┐
 sky.Sky ──► signals.simulate_voltages ──► correlator.correlate ──► pipeline.Visibilities ──► imaging
 (sources)   (per-antenna noise + delays)   F-step + X-step backend   (vis on u-v tracks)        dirty / CLEAN
                     ▲                                │                       │
              geometry.uvw                            │                calibration.stefcal
              layout.anneal_layout                    │                closure_phases
                                                      ▼
                                  ┌──── backends ─────────────────────────────┐
                                  │ numpy-loop · numpy-gemm                   │
                                  │ torch-cpu · torch-cuda                    │
                                  │ c-scalar · c-avx2 · c-avx2-omp (ctypes) ──┼──► csrc/fxcorr.c
                                  └───────────────────────────────────────────┘          ▲
                                                                                          │ same kernel
 ┌──────────────────────────── real-time streaming (C) ──────────────────────────────────┼────┐
 │ rx_sim (thread per receiver) ──TCP frames──► corrd: rx thread per receiver ─► ring buffers  │
 │   8-bit I/Q, seq numbers                     correlator thread: align seq ─► FFT ─► X-step ─┘
 │                                              ─► integrate ─► visibility dumps + stats         │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Layout of the repository

| path | contents |
|---|---|
| `src/radiotel/geometry.py` | ENU positions, baselines, (u, v, w), hour angles, campus footprint |
| `src/radiotel/sky.py` | `Observation` parameters, point-source `Sky`, analytic visibilities |
| `src/radiotel/signals.py` | voltage simulation, gain errors, 8-bit quantisation, two-tone test signals |
| `src/radiotel/phase.py` | two-signal phase estimators + Cramér–Rao bound |
| `src/radiotel/correlator/` | FX correlator front end + NumPy, Torch and C (ctypes) backends |
| `src/radiotel/pipeline.py` | `observe()` (voltages → correlator per snapshot) and `predict()` (analytic) |
| `src/radiotel/imaging.py` | gridding, dirty image/beam, Högbom CLEAN |
| `src/radiotel/calibration.py` | StEFCal gain solver, gain application, closure phases |
| `src/radiotel/layout.py` | u-v fill metric, random layouts, simulated annealing |
| `src/radiotel/plotting.py` | shared figure style |
| `csrc/fxcorr.c` | X-step kernels: scalar, cache-blocked AVX2+FMA, and the same with OpenMP over channel blocks |
| `csrc/fft.c` | radix-2 FFT for the streaming daemon |
| `csrc/stream/` | `rx_sim` (receivers), `corrd` (correlator daemon), wire format |
| `benchmarks/` | benchmark scripts, results CSVs, figure generation, Colab GPU notebook |
| `examples/` | four runnable walkthroughs that produce the README figures |
| `tests/` | pytest suite (geometry, signals, estimators, backends, imaging, calibration, layout) |
| `report/` | LaTeX technical report |
| `legacy/` | the original 2024 prototypes, untouched, with notes |

## Data layout conventions

* Voltages: `complex64` `(n_ant, n_samples)`.
* Spectra: `(n_ant, n_spec, nchan)`, C-contiguous. The C kernels take this
  array directly as interleaved `float` pairs, so there is no copy across the
  ctypes boundary.
* Visibilities: `(n_bl, nchan)` per correlator dump, with baselines in
  `numpy.triu_indices(n_ant, 1)` order. Pair *(i, j)* holds
  $\langle X_i X_j^*\rangle$. `pipeline.Visibilities.vis` stacks dumps as
  `(n_snapshots, n_bl, nchan)`.
* Channels are in FFT order. `Observation.channel_freqs()` gives their sky
  frequencies.

## The streaming daemon in detail

The original prototype (`legacy/client4.c`) had a receive thread per server
that called `recv()` straight into a shared buffer and only took the mutex
afterwards. The processing thread could read a half-written block, and frames
from the two receivers were never matched to each other. `corrd` fixes both
problems:

1. **Framing.** Every frame carries an antenna id and a sequence number
   (`frame.h`). Partial reads are completed with `MSG_WAITALL`.
2. **Ring buffer per receiver.** Each receive thread reads into private
   memory, then copies into slot `seq % ring` under the lock.
3. **Alignment.** The correlator thread waits until slot `seq` holds frame
   `seq` for *every* receiver before processing. Receivers that run ahead
   only fill future slots.
4. **Back-pressure instead of silent loss.** A receiver more than `ring`
   frames ahead blocks on a condition variable, so TCP flow control slows the
   sender. Gaps in sequence numbers are counted as `lost`. The paced mode of
   `rx_sim` counts *late* frames: when the correlator cannot keep up at the
   nominal rate, they show it directly.
5. **Plans and buffers allocated once.** FFT twiddles, bit-reversal tables
   and all buffers are created at start-up, unlike the legacy code, which
   rebuilt FFTW plans and called `malloc` for every block.

Output dumps (`rt_dump_header` + `complex64` visibilities) are read by
`benchmarks/stream_tools.py`. `bench_stream.sh` uses it to check `corrd`
against the NumPy correlator on the same quantised data.
