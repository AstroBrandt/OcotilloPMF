import numpy as np
import numpy.random as nr
import scipy.integrate as sint
import scipy.interpolate as si


class PMF:
    accObj = None
    IMF = None
    mmax = 100.0
    ml = 0.033

    def Chabrier05(self, m):
        m = np.asarray(m, dtype=float)
        bi = 0.740741 * (1.0 - self.mmax ** (-27.0 / 20.0))
        A1 = 1.0 / (2.851 + bi * 0.44956)
        A2 = 0.445956 * A1
        lo = A1 * np.exp(-((np.log10(m) - np.log10(0.2)) ** 2) / (2 * 0.55**2))
        hi = A2 * m ** (-1.35)
        result = np.where(m < 1.0, lo, hi)
        return result.item() if result.ndim == 0 else result

    def __init__(self, accObj, IMF=None, mmax=100.0, ml=0.033, seed=None):
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

    def IMFArr(self, m):
        return np.array([self.IMF(mi) for mi in m])

    def CIMF(self, ML=None, MU=None):
        if ML == None:
            ML = self.ml
        if MU == None:
            MU = self.mmax
        marr = np.logspace(np.log10(ML), np.log10(MU), int(1e4))
        integrand = self.IMFArr(marr) / marr
        cdist = sint.cumulative_trapezoid(integrand, x=marr, initial=0)
        f = si.interp1d(cdist, marr)
        return f

    def interpPhip2_mf(self, marr, mfarr, psip2, mi):
        if mi <= marr[0]:
            indx = 0
        elif mi >= marr[-1]:
            indx = len(marr) - 2
        else:
            try:
                indx = np.where(mi <= marr)[0][0] - 1
            except:
                print("ERROR: mi = ", mi)
                print("mmax = ", marr[-1])
                print("mmin = ", marr[0])
        mat0 = psip2[:, indx]
        mat1 = psip2[:, indx + 1]
        slp = (mat1 - mat0) / (marr[indx + 1] - marr[indx])
        dx = mi - marr[indx]
        return slp * dx + mat0

    def psip2(self, m, mf):
        tav = self.accObj.tmav(self.IMF, self.ml, self.mmax)
        return (self.IMF(mf) * self.accObj.tacc(m, mf)) / (tav)

    def PhiInvertSample(self, N=100):
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
