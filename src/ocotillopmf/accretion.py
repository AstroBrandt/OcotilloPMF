import numpy as np
import scipy.integrate as sint


class PowerLawAccrete:
    deltan1 = 0.0
    j = 1.0
    jf = 0.0
    m0 = 1e-5
    ml = 0.033
    mmax = 100.0

    def __init__(self, j, jf, m0, deltan1=0):
        self.j = j
        self.jf = jf
        self.m0 = m0
        self.deltan1 = deltan1

    def acc(self, m, mf):
        return (
            self.m0
            * (m / mf) ** self.j
            * mf**self.jf
            * (1.0 - self.deltan1 * (m / mf) ** (1.0 - self.j)) ** (0.5)
        )

    def tm(self, mf):
        return (mf ** (1.0 - self.jf) / ((1.0 - self.j) * self.m0)) * (1 + self.deltan1)

    def tacc(self, m, mf):
        # return (1.-self.j)*(m/mf)**(1.-self.j)*(1. - self.deltan1*(m/mf)**(1.-self.j))**(-0.5)*self.tm(mf)/(1.+self.deltan1)
        return m / self.acc(m, mf)

    def tmav(self, IMF, ML, MU):
        mfs = np.logspace(np.log10(ML), np.log10(MU), int(1e3))
        tmi = self.tm(mfs)
        imfi = np.array([IMF(mi) for mi in mfs])
        integrand = tmi * imfi
        return sint.trapezoid(integrand, x=np.log(mfs))
