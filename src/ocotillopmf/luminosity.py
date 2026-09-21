import numpy as np
import scipy.interpolate as si


class LuminosityObject:
    # PARAMETERS
    RSUN = 6.957e10
    LSUN = 3.848e33
    ALPHA = 0.39704170
    BETA = 8.52762600
    GAMMA = 0.00025546
    DELTA = 5.43288900
    EPSILON = 5.56357900
    ZETA = 0.78866060
    ETA = 0.00586685
    THETA = 1.71535900
    IOTA = 6.59778800
    KAPPA = 10.08855000
    LAMBDA = 1.01249500
    MU = 0.07490166
    NU = 0.01077422
    XI = 3.08223400
    UPSILON = 17.84778000
    PI = 0.00022582
    G = 6.67259e-8
    MSUN = 1.988e33

    accObj = None

    def lZAMS(self, m):
        LZAMS = (self.ALPHA * m**5.5 + self.BETA * m**11) / (
            self.GAMMA
            + m**3
            + self.DELTA * m**5
            + self.EPSILON * m**7
            + self.ZETA * m**8
            + self.ETA * m**9.5
        )
        return LZAMS

    def rZAMS(self, m):
        RZAMS = (
            self.THETA * m**2.5
            + self.IOTA * m**6.5
            + self.KAPPA * m**11
            + self.LAMBDA * m**19
            + self.MU * m**19.5
        ) / (
            self.NU
            + self.XI * m**2
            + self.UPSILON * m**8.5
            + m**18.5
            + self.PI * m**19.5
        )
        return RZAMS

    def LZAMS(self, m, mf):
        LZ = self.lZAMS(m) * self.LSUN
        return LZ

    def LACC(self, m, mf, r):  # All in solar units
        solarAcc_to_cgs = 6.305286e25  # to g/s
        LA = (self.G * (m * self.MSUN) * self.accObj.acc(m, mf) * solarAcc_to_cgs) / (
            r * self.RSUN
        )
        return LA

    def FUV_Frac(self, L, r):
        SIGMA = 5.6704e-5
        Teff = (L / (4.0 * np.pi * r * r * SIGMA)) ** (0.25)
        x_arr = [
            3.0,
            3.11111111111,
            3.22222222222,
            3.33333333333,
            3.44444444444,
            3.55555555556,
            3.66666666667,
            3.77777777778,
            3.88888888889,
            4.0,
            4.07142857143,
            4.14285714286,
            4.21428571429,
            4.28571428571,
            4.35714285714,
            4.42857142857,
            4.5,
            4.57142857143,
            4.64285714286,
            4.71428571429,
            4.78571428571,
            4.85714285714,
            4.92857142857,
            5.0,
            5.11111111111,
            5.22222222222,
            5.33333333333,
            5.44444444444,
            5.55555555556,
            5.66666666667,
            5.77777777778,
            5.88888888889,
            6.0,
        ]
        f_arr = [
            0.0,
            1.36770358435e-16,
            8.61653258138e-15,
            5.26258146666e-11,
            3.70098840631e-08,
            5.0568717347e-06,
            0.000195192270093,
            0.00284793494426,
            0.0197126224006,
            0.0772881806527,
            0.147011634854,
            0.240578127358,
            0.345971543508,
            0.444191207643,
            0.515303033114,
            0.54557500541,
            0.532074541579,
            0.482402999901,
            0.410347804342,
            0.330421281869,
            0.253964856801,
            0.187726633597,
            0.134328849088,
            0.0935685411832,
            0.0510859708182,
            0.0267964291578,
            0.0136432190711,
            0.00679353459533,
            0.00332675938709,
            0.00160869232546,
            0.000770502283946,
            0.000366365690472,
            0.000173237354368,
        ]
        frac = si.interp1d(x_arr, f_arr, kind="linear")
        try:
            ff = []
            for ti in Teff:
                if (np.log10(ti) < 3) or (np.log10(ti) > 6):
                    ff.append(0.0)
                else:
                    ff.append(frac(np.log10(ti)))
            return np.array(ff)
        except TypeError:
            if (np.log10(Teff) < 3) or (np.log10(Teff) > 6):
                return 0.0
            else:
                return frac(np.log10(Teff))

    def FUV_LUM(self, m, mf, r):
        lz = self.LZAMS(m, mf)
        la = self.LACC(m, mf, r * self.RSUN)
        lzfuv = lz * self.FUV_Frac(lz, r * self.RSUN)
        lafuv = la * self.FUV_Frac(la, r * self.RSUN)
        return lzfuv + lafuv

    def __init__(self, accObj):
        self.accObj = accObj
