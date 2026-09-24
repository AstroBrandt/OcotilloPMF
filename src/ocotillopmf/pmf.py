"""Sampling of the bivariate protostellar mass function (PMF)."""

from collections.abc import Callable

import numpy as np
import numpy.random as nr
import scipy.integrate as sint
import scipy.interpolate as si
from numpy.typing import ArrayLike

from .accretion import PowerLawAccrete


class PMF:
    """Sampler for the bivariate protostellar mass function.

    Given an accretion model and an initial mass function (IMF), samples
    the joint distribution of (current mass, final mass) for a
    population of protostars, following the PMF/PLF formalism of McKee &
    Offner (2010) and Offner & McKee (2011).

    Parameters
    ----------
    accObj : object
        Accretion model instance (e.g.
        [`PowerLawAccrete`][ocotillopmf.accretion.PowerLawAccrete]) exposing
        ``acc``, ``tacc``, and ``tmav`` methods.
    IMF : callable, optional
        Initial mass function, called as ``IMF(m)``. Defaults to
        [`Chabrier05`][ocotillopmf.pmf.PMF.Chabrier05].
    mmax : float, optional
        Maximum stellar mass, in Msun. Default is 100.0.
    ml : float, optional
        Minimum stellar mass, in Msun. Default is 0.033.
    seed : int, optional
        Seed for the global NumPy random number generator. If None, the
        generator is left unseeded (results are not reproducible).

    Attributes
    ----------
    accObj : object
        See Parameters.
    IMF : callable
        See Parameters.
    mmax : float
        See Parameters.
    ml : float
        See Parameters.
    """

    accObj: PowerLawAccrete | None = None
    IMF: Callable[[ArrayLike], float | np.ndarray] | None = None
    mmax = 100.0
    ml = 0.033

    def Chabrier05(self, m: ArrayLike) -> float | np.ndarray:
        """Chabrier (2005) system initial mass function.

        Parameters
        ----------
        m : float or array_like
            Stellar mass, in Msun.

        Returns
        -------
        float or ndarray
            Relative probability density at `m`. Used as the default
            `IMF` callable; not independently normalized.
        """
        m = np.asarray(m, dtype=float)
        bi = 0.740741 * (1.0 - self.mmax ** (-27.0 / 20.0))
        A1 = 1.0 / (2.851 + bi * 0.44956)
        A2 = 0.445956 * A1
        lo = A1 * np.exp(-((np.log10(m) - np.log10(0.2)) ** 2) / (2 * 0.55**2))
        hi = A2 * m ** (-1.35)
        result = np.where(m < 1.0, lo, hi)
        return result.item() if result.ndim == 0 else result

    def __init__(
        self,
        accObj: PowerLawAccrete,
        IMF: Callable[[ArrayLike], float | np.ndarray] | None = None,
        mmax: float = 100.0,
        ml: float = 0.033,
        seed: int | None = None,
    ) -> None:
        if seed == None:
            nr.seed()
        else:
            nr.seed(seed)
        if IMF == None:
            IMF = self.Chabrier05
        self.accObj = accObj
        self.IMF = IMF
        self.mmax = mmax
        self.ml = ml

    def IMFArr(self, m: ArrayLike) -> np.ndarray:
        """Vectorized IMF evaluation.

        Parameters
        ----------
        m : array_like
            Stellar masses, in Msun.

        Returns
        -------
        ndarray
            IMF value at each mass in `m`.
        """
        return np.array([self.IMF(mi) for mi in m])

    def CIMF(
        self, ML: float | None = None, MU: float | None = None
    ) -> Callable[[ArrayLike], np.ndarray]:
        """Inverse cumulative IMF, for inverse-transform sampling of the IMF alone.

        Parameters
        ----------
        ML : float, optional
            Lower mass bound, in Msun. Defaults to `self.ml`.
        MU : float, optional
            Upper mass bound, in Msun. Defaults to `self.mmax`.

        Returns
        -------
        callable
            Interpolating function mapping a cumulative probability in
            [0, 1] to a stellar mass, in Msun.
        """
        if ML == None:
            ML = self.ml
        if MU == None:
            MU = self.mmax
        marr = np.logspace(np.log10(ML), np.log10(MU), int(1e4))
        integrand = self.IMFArr(marr) / marr
        cdist = sint.cumulative_trapezoid(integrand, x=marr, initial=0)
        f = si.interp1d(cdist, marr)
        return f

    def interpPhip2_mf(
        self, marr: np.ndarray, mfarr: np.ndarray, psip2: np.ndarray, mi: float
    ) -> np.ndarray:
        """Linearly interpolate the bivariate PMF grid at a given current mass.

        Parameters
        ----------
        marr : ndarray
            Grid of current masses, in Msun.
        mfarr : ndarray
            Grid of final masses, in Msun.
        psip2 : ndarray
            Bivariate PMF evaluated on the `(mfarr, marr)` grid.
        mi : float
            Current mass at which to interpolate, in Msun.

        Returns
        -------
        ndarray
            PMF as a function of final mass, at `mi`.
        """
        if mi <= marr[0]:
            indx = 0
        elif mi >= marr[-1]:
            indx = len(marr) - 2
        else:
            try:
                indx = np.where(mi <= marr)[0][0] - 1
            except IndexError:
                raise ValueError(
                    f"Could not locate mi={mi} in marr (mmin={marr[0]}, mmax={marr[-1]}); "
                    "mi may be NaN or marr may not be sorted."
                ) from None

        mat0 = psip2[:, indx]
        mat1 = psip2[:, indx + 1]
        slp = (mat1 - mat0) / (marr[indx + 1] - marr[indx])
        dx = mi - marr[indx]
        return slp * dx + mat0

    def psip2(self, m: ArrayLike, mf: ArrayLike) -> float | np.ndarray:
        """Bivariate protostellar mass function.

        Parameters
        ----------
        m : float or array_like
            Current protostellar mass, in Msun.
        mf : float or array_like
            Final protostellar mass, in Msun.

        Returns
        -------
        float or ndarray
            Relative probability density of (m, mf).
        """
        tav = self.accObj.tmav(self.IMF, self.ml, self.mmax)
        return (self.IMF(mf) * self.accObj.tacc(m, mf)) / (tav)

    def PhiInvertSample(self, N: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Sample the bivariate (current mass, final mass) distribution.

        Uses inverse-transform sampling on a discretized version of the
        bivariate PMF to draw `N` independent (m, mf) pairs.

        Parameters
        ----------
        N : int, optional
            Number of protostars to sample. Default is 100.

        Returns
        -------
        m : ndarray
            Sampled current masses, in Msun.
        mf : ndarray
            Sampled final masses, in Msun.
        """
        # Calculate the analytic bivariate on a massres^2 grid
        massres = 512
        mi = np.logspace(np.log10(self.ml), np.log10(self.mmax), massres)
        mfi = np.logspace(np.log10(self.ml), np.log10(self.mmax), massres)
        tav = self.accObj.tmav(self.IMF, self.ml, self.mmax)
        M, MF = np.meshgrid(mi, mfi)
        with np.errstate(invalid="ignore", divide="ignore"):
            TACC = self.accObj.tacc(M, MF)
        IMF_col = self.IMFArr(mfi)[:, None]
        psip2arr = np.where(M < MF, IMF_col * TACC / tav, 0.0)

        # Calculate the marginalization over final masses (essentially the PMF) and
        # ensure its normalized to 1
        psip2ymarg = sint.trapezoid(psip2arr, x=np.log(mfi), axis=0)
        psip2ymarg /= sint.trapezoid(psip2ymarg, x=np.log(mi))
        # Calculate the cumulative for inverse-transform sampling of the current mass
        CPSIX = sint.cumulative_trapezoid(psip2ymarg, x=np.log(mi), initial=0)

        mis = []
        mfs = []
        # N = number of stars to sample
        for ip in range(N):
            # Pull random value, use it to get the current mass
            rnd1 = nr.uniform(0, 1)
            mrnd = np.interp(rnd1, CPSIX, mi)
            # Given the current mass, calculate the conditional probabilty for the final mass
            # This is a sharp function that has a discontinuity at m = mf
            # The mask filters out the unphysical discontinuity, then it has to be rescaled to
            # ensure it integrates to 1. The rescaling needs to be done because of the finite grid
            # size making the initial discontinuity not as sharp
            psix = self.interpPhip2_mf(mi, mfi, psip2arr, mrnd) / np.interp(
                mrnd, mi, psip2ymarg
            )
            indx0 = np.where(psix > 0)[0][0]
            mfi_n = mfi[indx0:]
            psix_n = psix[indx0:]
            # If mrnd falls in the last grid cell, only one (or zero) mf grid points can
            # satisfy mf > mrnd, so there's no distribution left to build (trapezoid of a
            # single point is 0). In that case the final mass is effectively pinned to the
            # one surviving grid point (mmax).
            if len(mfi_n) < 2:
                mis.append(mrnd)
                mfs.append(mfi_n[-1])
                continue
            psix_n = psix_n / sint.trapezoid(psix_n, x=np.log(mfi_n))
            CPSIY = sint.cumulative_trapezoid(psix_n, x=np.log(mfi_n), initial=0)
            rnd2 = nr.uniform(0, 1)
            mfrnd = np.interp(rnd2, CPSIY, mfi_n)
            mis.append(mrnd)
            mfs.append(mfrnd)

        return np.array(mis), np.array(mfs)

    def calcPMF(
        self, ML: float = 0.04, MU: float = 3.0, res: int = 2**8
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate a semi-analytic PMF for a given upper and lower mass range.

        Numerically integrates the equation for the PMF, rather than discretely sampling the bivariation function.

        Parameters
        ----------
        ML : float, optional
            Lower limit of the mass function. Default is 0.04.
        MU : float, optional
            Upper limit of the mass function. Default is 3.0.
        res : int, optional
            Size of the returning arrays. Default is 256.

        Returns
        -------
        m : ndarray
            Mass array of the functional, psi(m)
        PSIM : ndarray
            PMF as a function of mass.
        """
        m = np.logspace(np.log10(ML), np.log10(MU), res)
        PSIM = []
        for mi in m:
            mf = np.logspace(np.log10(max(ML, mi)), np.log10(MU), res)
            # tacc(m=mf) is singular for tapered accretion (accretion rate -> 0 as m -> mf),
            # so exclude that boundary point, mirroring PhiInvertSample's strict mi < mfi mask.
            mf = mf[mf > mi]
            if len(mf) < 2:
                PSIM.append(0.0)
                continue
            integrand = self.IMFArr(mf) * self.accObj.tacc(mi, mf)
            PSIM.append(sint.trapezoid(integrand, x=np.log(mf)))
        PSIM = np.array(PSIM) / self.accObj.tmav(self.IMF, ML, MU)
        return m, PSIM

    def synthesisClusterStatistic(
        self,
        Nproto: ArrayLike,
        Nsamp: ArrayLike | None = None,
        funcQuantity: Callable[[np.ndarray, np.ndarray, np.ndarray], ArrayLike]
        | None = None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Mean and standard deviation of a cluster-integrated quantity.

        For each cluster size in `Nproto`, draws `Nsamp` independent
        clusters from the PMF, evaluates `funcQuantity` for every
        protostar, sums it over each cluster, and computes the mean and
        standard deviation of that per-cluster sum across the draws.

        Parameters
        ----------
        Nproto : int or array_like
            Number of protostars per cluster. May be a scalar or an array of cluster sizes.
        Nsamp : int, array_like, or None, optional
            Number of cluster draws per `Nproto` value. If None
            (default), uses ``max(10, 1e5 / Nproto)`` for each `Nproto`.
            If `Nproto` is an array, `Nsamp` may be a scalar (applied to
            every `Nproto`), an array matching the size of `Nproto`, or
            None.
        funcQuantity : callable
            Function called as ``funcQuantity(m, mf, mdot)``, with
            arrays of current mass, final mass, and accretion rate for a
            cluster's protostars (in Msun and Msun/yr). Must return an
            array of per-protostar quantities, elementwise in its
            inputs, which are summed to give the cluster total.

        Returns
        -------
        mean : float or ndarray
            Mean of the per-cluster summed quantity, for each `Nproto`.
        std : float or ndarray
            Standard deviation of the per-cluster summed quantity, for
            each `Nproto`.
        """
        if funcQuantity is None:
            raise ValueError("funcQuantity must be provided.")

        scalar_input = np.ndim(Nproto) == 0
        Nproto_arr = np.atleast_1d(np.asarray(Nproto)).astype(int)

        if Nsamp is None:
            Nsamp_arr = np.maximum(10, (1e5 / Nproto_arr.astype(float))).astype(int)
        else:
            Nsamp_arr = np.atleast_1d(np.asarray(Nsamp))
            if Nsamp_arr.size == 1:
                Nsamp_arr = np.full(Nproto_arr.shape, Nsamp_arr[0])
            elif Nsamp_arr.size != Nproto_arr.size:
                raise ValueError(
                    "Nsamp must be a scalar, None, or match the size of Nproto."
                )
            Nsamp_arr = Nsamp_arr.astype(int)

        meanArr = []
        stdArr = []
        for Nprotoi, Nsampi in zip(Nproto_arr, Nsamp_arr):
            mbig_arr, mfbig_arr = self.PhiInvertSample(N=int(Nprotoi * Nsampi))
            mdotbig_arr = self.accObj.acc(mbig_arr, mfbig_arr)
            qbig_arr = funcQuantity(mbig_arr, mfbig_arr, mdotbig_arr)

            sumArr = np.asarray(qbig_arr).reshape(Nsampi, Nprotoi).sum(axis=1)
            meanArr.append(np.mean(sumArr))
            stdArr.append(np.std(sumArr))

        meanArr = np.array(meanArr)
        stdArr = np.array(stdArr)

        if scalar_input:
            return meanArr[0], stdArr[0]
        return meanArr, stdArr

    def get_Ns(self, mmax: float, ML: float = 0.04, MU: float = 3.0) -> float:
        """Calculate the expected number of stars in a cluster, given a maximal sampled mass.

        Uses the PMF to solve for the expected number of protostars needed to be in a cluster to get a star of mass mmax, given the PMF mass limits

        Parameters
        ----------
        mmax : float
            Maximum protostar mass in the cluster
        ML : float, optional
            Lower limit of the mass function. Default is 0.04.
        MU : float, optional
            Upper limit of the mass function. Default is 3.0.

        Returns
        -------
        Ns : float
            Expected number of stars needed in the cluster to have at least 1 protostar with mass mmax.
        """
        if mmax > MU:
            raise ValueError("MU must be higher than mmax.")
        marr, psim = self.calcPMF(ML, MU)
        psim_func = si.interp1d(np.log(marr), psim)
        mfarr = np.logspace(np.log10(mmax), np.log10(MU), 2**7)
        psi_samp = psim_func(np.log(mfarr))
        integ = sint.trapezoid(psi_samp, x=np.log(mfarr))
        return 1.0 / integ
