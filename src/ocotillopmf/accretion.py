"""Steady-state protostellar accretion models."""

from collections.abc import Callable

import numpy as np
import scipy.integrate as sint
from numpy.typing import ArrayLike


class PowerLawAccrete:
    """Tapered power-law accretion model for a protostar.

    Implements the steady-state accretion law relating a protostar's
    instantaneous accretion rate to its current mass `m` and final mass
    `mf`, following the formalism of McKee & Offner (2010) and Offner &
    McKee (2011).

    Parameters
    ----------
    j : float
        Power-law index of the accretion rate's dependence on `m/mf`.
    jf : float
        Power-law index of the accretion rate's dependence on `mf`.
    m0 : float
        Normalization of the accretion rate, in Msun/yr.
    deltan1 : float, optional
        Tapering parameter that drives the accretion rate to zero as
        `m -> mf`. Default is 0 (no tapering).

    Attributes
    ----------
    j : float
        See Parameters.
    jf : float
        See Parameters.
    m0 : float
        See Parameters.
    deltan1 : float
        See Parameters.
    """

    deltan1 = 0.0
    j = 1.0
    jf = 0.0
    m0 = 1e-5
    ml = 0.033
    mmax = 100.0

    def __init__(self, j: float, jf: float, m0: float, deltan1: float = 0) -> None:
        self.j = j
        self.jf = jf
        self.m0 = m0
        self.deltan1 = deltan1

    def acc(self, m: ArrayLike, mf: ArrayLike) -> float | np.ndarray:
        """Instantaneous mass accretion rate.

        Parameters
        ----------
        m : float or array_like
            Current protostellar mass, in Msun.
        mf : float or array_like
            Final protostellar mass, in Msun.

        Returns
        -------
        float or ndarray
            Accretion rate dm/dt, in Msun/yr.
        """
        return (
            self.m0
            * (m / mf) ** self.j
            * mf**self.jf
            * (1.0 - self.deltan1 * (m / mf) ** (1.0 - self.j)) ** (0.5)
        )

    def tm(self, mf: ArrayLike) -> float | np.ndarray:
        """Total formation timescale for a star of final mass `mf`.

        Parameters
        ----------
        mf : float or array_like
            Final protostellar mass, in Msun.

        Returns
        -------
        float or ndarray
            Formation timescale, in yr.
        """
        return (mf ** (1.0 - self.jf) / ((1.0 - self.j) * self.m0)) * (1 + self.deltan1)

    def tacc(self, m: ArrayLike, mf: ArrayLike) -> float | np.ndarray:
        """Instantaneous accretion timescale, ``m / acc(m, mf)``.

        Parameters
        ----------
        m : float or array_like
            Current protostellar mass, in Msun.
        mf : float or array_like
            Final protostellar mass, in Msun.

        Returns
        -------
        float or ndarray
            Accretion timescale, in yr.
        """
        # return (1.-self.j)*(m/mf)**(1.-self.j)*(1. - self.deltan1*(m/mf)**(1.-self.j))**(-0.5)*self.tm(mf)/(1.+self.deltan1)
        return m / self.acc(m, mf)

    def tmav(self, IMF: Callable[[float], float], ML: float, MU: float) -> float:
        """IMF-averaged formation timescale over a mass range.

        Parameters
        ----------
        IMF : callable
            Initial mass function, called as ``IMF(m)`` for a final mass
            `m`.
        ML : float
            Lower mass bound of the integral, in Msun.
        MU : float
            Upper mass bound of the integral, in Msun.

        Returns
        -------
        float
            IMF-weighted average formation timescale, in yr.
        """
        mfs = np.logspace(np.log10(ML), np.log10(MU), int(1e3))
        tmi = self.tm(mfs)
        imfi = np.array([IMF(mi) for mi in mfs])
        integrand = tmi * imfi
        return sint.trapezoid(integrand, x=np.log(mfs))
