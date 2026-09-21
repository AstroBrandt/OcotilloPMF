"""Adapted from examples/protoclusterGen.ipynb: checks that PhiInvertSample's bivariate sampling converges onto the semi-analytic PMF as N grows."""

import numpy as np
import scipy.integrate as sint

from ocotillopmf import PMF, PowerLawAccrete

BINS = np.linspace(-4, 5, 25)


def analytic_pmf(plaw, imf, ml, mmax, n_grid=500):
    m = np.logspace(np.log10(ml), np.log10(mmax), n_grid)
    psim = []
    for mi in m:
        mf = np.logspace(np.log10(max(ml, mi)), np.log10(mmax), n_grid)
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
    hist, edges = np.histogram(np.log(m_arr), bins=BINS, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    analytic_at_centers = np.interp(centers, np.log(marr), psim)
    return np.sqrt(np.mean((hist - analytic_at_centers) ** 2))


def test_phi_invert_sample_converges_to_analytic_pmf():
    plaw = PowerLawAccrete(0.5, 0.75, 3.6e-5, deltan1=1.0)
    pmf = PMF(plaw, seed=42)
    marr, psim = analytic_pmf(plaw, pmf.IMF, pmf.ml, pmf.mmax)

    errors = []
    for n in (10, 100, 1000, 10000):
        m_arr, _ = pmf.PhiInvertSample(N=n)
        errors.append(sampling_error(m_arr, marr, psim))

    # RMS error against the analytic curve should shrink monotonically as
    # the sample size grows...
    assert all(e2 < e1 for e1, e2 in zip(errors, errors[1:]))
    # ...and the largest sample should land close to the analytic solution.
    assert errors[-1] < 0.06
