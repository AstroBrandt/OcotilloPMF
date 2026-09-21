"""Adapted from examples/protoclusterGen.ipynb: checks that PhiInvertSample's
bivariate sampling converges onto the semi-analytic PMF as N grows."""

import platform
from itertools import pairwise

import numpy as np
import scipy
import scipy.integrate as sint

from ocotillopmf import PMF, PowerLawAccrete

SAMPLE_SIZES = (100, 1000, 10000, 100000)


def analytic_pmf(plaw, imf, ml, mmax, n_grid=500):
    m = np.logspace(np.log10(ml), np.log10(mmax), n_grid)
    psim = []
    for mi in m:
        mf = np.logspace(np.log10(max(ml, mi)), np.log10(mmax), n_grid)
        # tacc(m=mf) is singular for tapered accretion (accretion rate -> 0 as m -> mf),
        # so exclude that boundary point, mirroring PhiInvertSample's strict mi < mfi mask.
        mf = mf[mf > mi]
        if len(mf) < 2:
            psim.append(0.0)
            continue
        integrand = np.array([imf(mfi) for mfi in mf]) * plaw.tacc(mi, mf)
        psim.append(sint.trapezoid(integrand / mf, x=mf))
    psim = np.array(psim) / plaw.tmav(imf, ml, mmax)
    psim /= sint.trapezoid(psim, x=np.log(m))
    return m, psim


def sampling_error(m_arr, marr, psim):
    # 'auto' bins adapt to where the sample actually concentrates, instead of
    # a fixed range that's mostly empty and lets a handful of stray counts
    # dominate the RMS comparison.
    hist, edges = np.histogram(np.log(m_arr), bins="auto", density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    analytic_at_centers = np.interp(centers, np.log(marr), psim)
    return np.sqrt(np.mean((hist - analytic_at_centers) ** 2))


def test_phi_invert_sample_converges_to_analytic_pmf():
    print(
        f"\nnumpy {np.__version__}, scipy {scipy.__version__}, "
        f"platform {platform.platform()}"
    )

    plaw = PowerLawAccrete(0.5, 0.75, 3.6e-5, deltan1=1.0)
    pmf = PMF(plaw, seed=42)
    marr, psim = analytic_pmf(plaw, pmf.IMF, pmf.ml, pmf.mmax)

    # Each star consumes either one or two draws from the global RNG stream
    # (PhiInvertSample's `continue` branch skips the second draw), so a tiny
    # floating-point difference between platforms can flip that branch for a
    # single star and desync every draw after it. Re-seeding before each N
    # keeps the runs independent instead of letting an unrelated smaller-N
    # run perturb a later one.
    errors = []
    for n in SAMPLE_SIZES:
        pmf = PMF(plaw, seed=42)
        m_arr, _ = pmf.PhiInvertSample(N=n)
        errors.append(sampling_error(m_arr, marr, psim))

    print(f"errors for N={SAMPLE_SIZES}: {errors}")

    # RMS error against the analytic curve should shrink monotonically as N grows.
    assert all(e2 < e1 for e1, e2 in pairwise(errors)), errors
    # The largest sample should land close to the analytic solution.
    assert errors[-1] < 0.01, errors
