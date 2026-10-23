/*
 * X-step of an FX correlator.
 *
 * spectra: n_ant * n_spec * nchan complex floats (interleaved re, im), laid out
 *          as spectra[ant][spec][chan] - i.e. a C-contiguous numpy complex64
 *          array of shape (n_ant, n_spec, nchan).
 * vis:     n_bl * nchan complex floats, n_bl = n_ant * (n_ant - 1) / 2,
 *          baselines ordered like numpy.triu_indices(n_ant, 1).
 *
 * vis[b][c] = (1 / n_spec) * sum_t spectra[i][t][c] * conj(spectra[j][t][c])
 */
#ifndef FXCORR_H
#define FXCORR_H

#ifdef __cplusplus
extern "C" {
#endif

int fx_n_baselines(int n_ant);
int fx_cpu_has_avx2(void);
int fx_max_threads(void);
void fx_set_threads(int n);

void fx_xstep_scalar(const float *spectra, int n_ant, int n_spec, int nchan, float *vis);
void fx_xstep_avx2(const float *spectra, int n_ant, int n_spec, int nchan, float *vis);
void fx_xstep_avx2_omp(const float *spectra, int n_ant, int n_spec, int nchan, float *vis);

#ifdef __cplusplus
}
#endif

#endif
