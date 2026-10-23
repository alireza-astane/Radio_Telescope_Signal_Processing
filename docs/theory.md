# Theory: from phase differences to sky images

This page explains the physics and signal processing behind each stage of the
pipeline, with pointers to the code that implements it.

## 1. Why phase differences?

A distant radio source is effectively at infinity, so its wavefront is flat.
Two receivers separated by a baseline vector $\mathbf b$ see the same
wavefront with a **geometric delay**

$$\tau_g = \frac{\mathbf b \cdot \hat{\mathbf s}}{c},$$

where $\hat{\mathbf s}$ is the unit vector towards the source. At frequency
$\nu$ the delay appears as a phase difference $2\pi\nu\tau_g$. That phase is
the measurement: it encodes where the source is, to a precision set by
$\lambda / |\mathbf b|$. A 300 m baseline at 21 cm resolves about 2.4 arcmin,
while a single 2 m dish resolves about 6°.

With $N$ receivers every pair forms a baseline, giving
$N(N-1)/2$ simultaneous phase measurements (`geometry.baselines`).

## 2. Signals are noise, so the phase is in the correlation

Astronomical sources emit stationary Gaussian noise rather than sine waves,
so a single receiver's voltage has no well-defined phase. What *is*
well-defined is the **cross-correlation** of two receivers. In the frequency
domain this is the visibility

$$V_{ij}(\nu) = \langle X_i(\nu)\, X_j^*(\nu) \rangle,$$

averaged over time. The van Cittert–Zernike theorem relates it to the sky
brightness $I(l, m)$:

$$V(u, v, w) = \iint I(l, m)\, e^{-2\pi i\,(ul + vm + w(n-1))}\, \frac{dl\,dm}{n},$$

where $(u, v, w) = \mathbf b/\lambda$ in a frame whose $w$-axis points at the
phase centre, and $(l, m, n)$ are direction cosines. **Each baseline samples
one Fourier component of the sky.** The amplitude of $V$ is the correlated
flux and its phase is the phase difference from §1.

`signals.simulate_voltages` generates exactly this situation. Each source is
complex Gaussian noise, delayed per antenna by
$\tau_a = -(u_a l + v_a m + w_a(n-1))/c$. The delay is applied as an exact
phase ramp $e^{-2\pi i(\nu_0 + f)\tau_a}$ in the frequency domain, so
fractional-sample delays are correct. Independent receiver noise and
complex gain errors are then added.

## 3. Estimating a phase difference: the original question

The original project compared ways to get the phase difference of two
signals (`phase.py`, `benchmarks/bench_phase.py`):

| method | idea | weakness |
|---|---|---|
| FFT peak | $\arg X_2[k^*] X_1^*[k^*]$ at the strongest bin | single tone only |
| Hilbert + unwrap | mean of unwrapped instantaneous-phase difference | unwrapping fails at low SNR (one slip biases the whole mean) |
| Hilbert cross | $\arg \sum a_2 a_1^*$ of the analytic signals | single tone only |
| cross-correlation lag | peak lag × $2\pi f / f_s$ | needs the frequency; whole-period ambiguity (the legacy version also quantised to whole samples) |

For a tone in white noise, no unbiased estimator can beat the
**Cramér–Rao bound**
$\operatorname{var}(\hat\varphi_2 - \hat\varphi_1) \ge 2/(N \cdot \mathrm{SNR})$,
with $\mathrm{SNR} = A^2/2\sigma^2$. The benchmark shows that the cross-product
estimators reach the bound, while Hilbert+unwrap breaks down below about 0 dB.

None of these works on real sky signals, which are broadband noise. The
generalisation that does is to channelise first and take the cross-product
per channel. That is the FX correlator.

## 4. The FX correlator

* **F-step:** split each receiver's stream into blocks of `nchan` samples and
  FFT each block (unitary normalisation, so $|V| =$ flux).
  Cost: $O(N \cdot S \log n_{chan})$ for $S$ samples per receiver.
* **X-step:** for every baseline and channel, accumulate
  $X_i X_j^*$ over the blocks. Cost: $O(N^2 S / 2)$ complex
  multiply-accumulates.

The X-step dominates for large arrays, which is the "combinatorial explosion"
the original project anticipated. It has two efficient forms:

1. **Cache-blocked, vectorised over channels** (`csrc/fxcorr.c`). Four complex
   numbers fit in one AVX2 register. Instead of shuffling every product, the
   kernel accumulates $P = a \odot b$ and $Q = a \odot \mathrm{swap}(b)$ with
   fused multiply-adds and forms $\mathrm{Re} = P_{even} + P_{odd}$,
   $\mathrm{Im} = Q_{odd} - Q_{even}$ once per channel block.

   The memory layout matters more than the arithmetic. Walking baseline by
   baseline re-reads every antenna $N-1$ times from DRAM with a large stride,
   and the first version of the kernel stalled at about 5 GFLOP/s with no gain
   from more threads. The final kernel works on blocks of 8 channels. For each
   block it copies all antennas into a contiguous buffer (one 64-byte cache
   line per antenna and time step), forms every baseline from that buffer
   while it sits in L1/L2, and register-tiles two baselines per load of
   antenna $i$. OpenMP parallelises over channel blocks.
2. **As matrix products** (`numpy_fx.xstep_gemm`, `torch_fx`). For each
   channel, $R = S S^H$ with $S$ the $N \times n_{spec}$ matrix of spectra.
   This is how GPU correlators use tensor cores.

The alternative XF ordering (correlate in time, then FFT the lag spectrum)
costs $O(N^2 S \cdot n_{lags})$ and is only used in specialised hardware.

## 5. Earth-rotation synthesis

A baseline fixed on the ground traces an ellipse in the $(u, v)$ plane as the
Earth rotates (`geometry.uvw`, ported from the original `getV` notebook code):

$$\begin{pmatrix}u\\v\\w\end{pmatrix} =
\begin{pmatrix}\sin H & \cos H & 0\\ -\sin\delta\cos H & \sin\delta\sin H & \cos\delta\\
\cos\delta\cos H & -\cos\delta\sin H & \sin\delta\end{pmatrix}
\begin{pmatrix}X\\Y\\Z\end{pmatrix},$$

with $H$ the hour angle, $\delta$ the declination and $(X, Y, Z)$ the baseline
in the equatorial frame obtained from local east-north-up at latitude $\phi$.
Eight hours of observing turn 15 baselines into 15 arcs of Fourier coverage.

## 6. Imaging and deconvolution

Gridding the visibilities (`imaging.grid`) and inverse-FFTing gives the
**dirty image**: the true sky convolved with the **dirty beam** (the PSF),
which is the Fourier transform of the sampling pattern. Sparse coverage
produces strong sidelobes. **Högbom CLEAN** (`imaging.hogbom_clean`)
repeatedly finds the brightest residual pixel, records a fraction of it as a
point component, and subtracts the correspondingly shifted PSF. The
components are then convolved with a Gaussian "clean beam".

Simplifications, documented in the code:

* nearest-neighbour gridding on an oversampled grid (`pad = 4`) instead of
  convolutional gridding;
* no w-term correction, which is valid for fields within about 0.5° on these
  baselines;
* no primary-beam correction.

## 7. Calibration and closure phases

Each receiver multiplies its signal by an unknown complex gain $g_k$
(cable length, amplifier, oscillator), so
$V^{obs}_{ij} = g_i g_j^* V_{ij}$. There are $N$ unknowns but $N(N-1)/2$
measurements, so observing a known calibrator lets **StEFCal**
(`calibration.stefcal`, Salvini & Wijnholds 2014) solve them by alternating
least squares. **Closure phases** around any triangle,
$\arg V_{ij} + \arg V_{jk} - \arg V_{ik}$, cancel every antenna phase
exactly. They are a gain-free observable, which makes them useful for
checking a calibration.

## 8. Choosing where the receivers go

Better $(u,v)$ coverage means a cleaner PSF. `layout.anneal_layout`
maximises the fraction of u-v cells that the Earth-rotation tracks visit,
moving one receiver at a time inside the campus footprint with a minimum
spacing. It replaces the original notebook score, which counted pixels in a
matplotlib plot, with the same quantity computed directly. This makes it
fast enough for thousands of annealing steps.

## References

* Thompson, Moran & Swenson, *Interferometry and Synthesis in Radio Astronomy*, 3rd ed. (2017).
* Taylor, Carilli & Perley (eds.), *Synthesis Imaging in Radio Astronomy II*, ASP Conf. Ser. 180 (1999).
* Högbom, "Aperture synthesis with a non-regular distribution of interferometer baselines", A&AS 15, 417 (1974).
* Salvini & Wijnholds, "Fast gain calibration in radio astronomy using alternating direction implicit methods: analysis and applications", A&A 571, A97 (2014).
* Kay, *Fundamentals of Statistical Signal Processing, Vol. I: Estimation Theory* (1993).
* Clark et al., "Accelerating radio astronomy cross-correlation with graphics processing units", IJHPCA 27(2) (2013).
