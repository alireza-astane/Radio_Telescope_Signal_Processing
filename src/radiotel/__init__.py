"""radiotel: a small radio-interferometer simulator and FX correlator.

Pipeline: sky model -> per-antenna voltages (``signals``) -> FX correlator
(``correlator``) -> visibilities on Earth-rotation u-v tracks (``geometry``)
-> calibration (``calibration``) -> dirty image + CLEAN (``imaging``);
plus array-layout optimisation (``layout``) and two-signal phase
estimators (``phase``).
"""

__version__ = "1.0.0"
