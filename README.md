<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AstroBrandt/OcotilloPMF/main/docs/assets/ocotillopmf-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/AstroBrandt/OcotilloPMF/main/docs/assets/ocotillopmf-logo-light.svg">
    <img alt="OcotilloPMF" src="https://raw.githubusercontent.com/AstroBrandt/OcotilloPMF/main/docs/assets/ocotillopmf-logo-light.svg" height="110">
  </picture>
</p>

A Python package for sampling the protostellar mass function (PMF) — the
joint distribution of current and final stellar mass during star formation —
under a given accretion model, plus tools for computing the resulting
accretion and photospheric luminosities.

## Installation

```bash
pip install ocotillopmf
```

To also run the example notebooks (requires matplotlib, seaborn, jupyter):

```bash
pip install "ocotillopmf[examples]"
```

## Quickstart

```python
from ocotillopmf import PowerLawAccrete, PMF

# Define a tapered turbulent-core accretion model: PowerLawAccrete(j, jf, m0, deltan1)
accretion = PowerLawAccrete(0.5, 0.75, 3.6e-5, deltan1=1.0)

# Build a PMF sampler on top of it (defaults to a Chabrier 2005 IMF)
pmf = PMF(accretion)

# Sample the bivariate (current mass, final mass) distribution for N protostars
m, mf = pmf.PhiInvertSample(N=10_000)

# Instantaneous accretion rate for each sampled star, in Msun/yr
mdot = accretion.acc(m, mf)
```

See [examples/protoclusterGen.ipynb](examples/protoclusterGen.ipynb) for a
full walkthrough, including luminosity calculations via `LuminosityObject`
and convergence checks on the sampler.

## License

TBD

## Author

Brandt (brandt.gaches@uni-due.de)
