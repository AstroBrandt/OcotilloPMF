"""Spatial sampling methods to produce physical clusters, also produce clouds if desired."""

import numpy as np
import numpy.random as nr
import scipy.integrate as sint
import scipy.interpolate as si
import scipy.fft as sfft
import scipy.special as ssp
import scipy.spatial.distance as sdist

from FyeldGenerator import generate_field


class Spatial:
    """Generator for fractal gas clouds and the spatial distribution of star clusters.

    Builds log-normal fractional Brownian motion (fBm) density fields with
    a given fractal dimension or Hurst exponent, and uses them either
    directly as model gas clouds or as a probability density from which
    protostar positions are drawn. Cluster positions can optionally be mass segregated
    following Baumgardt et al. (2008), as implemented in McLuster
    (Kuepper et al. 2011).

    Parameters
    ----------
    seed : int, optional
        Seed stored on the object. Every call to :meth:`makeFBM` (and so
        :meth:`makeCloudFBM` and :meth:`makeStellarCluster`) draws from a
        fresh generator seeded with it, so repeated calls reproduce the
        same field, and a cloud and a cluster made from the same object
        share the same structure. The global NumPy random state is not
        modified. Individual calls can override it with `overSeed`. If
        None, each call draws from a freshly, randomly seeded generator,
        so results are not reproducible and separate calls do not share
        structure.
    """

    def __init__(self, seed=None):
        self.seed = seed

    def _rng(self, overSeed=None):
        """Fresh random number generator seeded with `overSeed`, or the object's `seed`.

        Uses NumPy's legacy RandomState (Mersenne Twister), the same
        algorithm as ``numpy.random.seed``, so a given seed reproduces the
        same draws as ``numpy.random.seed`` with that seed, while leaving
        the global NumPy state untouched.
        A seed of None seeds the generator randomly from the operating system.
        """
        if overSeed == None:
            return nr.RandomState(self.seed)
        return nr.RandomState(overSeed)

    def makeFBM(
        self,
        ndim=3,
        D=None,
        H=None,
        L=1.0,
        nres=128,
        expon=True,
        scale=1,
        log_offset=0,
        overSeed=None,
        rng=None,
    ):
        """Generate a periodic fractional Brownian motion (fBm) field.

        A Gaussian random field with power-law power spectrum
        P(k) ~ k^(-beta) is generated on a regular grid and normalized to
        unit standard deviation. If `expon` is True the field is
        exponentiated, giving a log-normal (log-fBm) field.

        The spectral index is set by either the Hurst exponent, beta =
        ndim + 2 H, or the fractal dimension, beta = 2 (4 - D) (Stutzki et
        al. 1998). The `D` relation is that of a 2D map, so `D` is the
        fractal dimension of the projected field. A true fBm requires
        ndim <= beta <= ndim + 2, i.e. 0 <= H <= 1; a warning is printed
        outside this range, since the field is then no longer self-affine.
        For ``ndim=3`` this means 1.5 <= `D` <= 2.5.

        Parameters
        ----------
        ndim : int, optional
            Number of spatial dimensions. Default is 3.
        D : float, optional
            Fractal dimension of the projected field. Cannot be given
            together with `H`. If neither is given, defaults to 2.4.
        H : float, optional
            Hurst exponent, in [0, 1], setting the roughness of the field
            in any dimension. H = 1/3 in 3D (beta = 11/3) gives a
            Kolmogorov-like log-density spectrum. Cannot be given together
            with `D`.
        L : float, optional
            Side length of the (cubic) box. The grid spans [-L/2, L/2]
            along each axis. Default is 1.0.
        nres : int, optional
            Number of grid cells along each axis. Default is 128.
        expon : bool, optional
            If True, return exp(`log_offset` + `scale` * field) instead
            of the Gaussian field. Default is True.
        scale : float, optional
            Standard deviation of the log of the field when `expon` is
            True. Default is 1.
        log_offset : float, optional
            Mean of the log of the field when `expon` is True. Default
            is 0.
        overSeed : int, optional
            Seed to use for this call instead of the object's `seed`.
            The stored `seed` is not changed.
        rng : numpy.random.RandomState, optional
            Generator to draw from, overriding `seed` and `overSeed`.
            Used internally so that later draws (e.g. star positions)
            continue the same random stream. If None, a new generator is
            seeded from `overSeed` or `seed`. The global NumPy random
            state is never modified.

        Returns
        -------
        xgrid : ndarray
            Cell edges, of shape (ndim, nres + 1); ``xgrid[d]`` holds
            the edges along axis ``d``.
        field : ndarray
            The field, of shape (nres,) * ndim, indexed so that array
            axis ``d`` runs along spatial axis ``d``. Periodic along
            every axis.

        Raises
        ------
        ValueError
            If both `D` and `H` are given.
        """
        if rng is None:
            rng = self._rng(overSeed)

        # Helper that generates power-law power spectrum
        def Pkgen(n):
            def Pk(k):
                return np.power(k, -n)

            return Pk

        # Draw samples from a normal distribution
        def distrib(shape):
            a = rng.normal(loc=0, scale=1, size=shape)
            b = rng.normal(loc=0, scale=1, size=shape)
            return a + 1j * b

        if D != None and H != None:
            raise ValueError(
                "[OcotilloPMF error] Give either the fractal dimension D or the Hurst exponent H, not both."
            )

        if H != None:
            specIndex = ndim + 2.0 * H
        else:
            if D == None:
                D = 2.4
            specIndex = 2.0 * (4.0 - D)

        if (specIndex < ndim) or (specIndex > ndim + 2):
            print(
                f"[OcotilloPMF WARNING]: Spectral index {specIndex:g} is outside [ndim, ndim + 2] (H outside [0, 1]), so the field is not a true fBm. Will still run, but likely not intended! "
            )

        shape = tuple([nres for i in range(ndim)])
        L2 = L / 2.0

        xgrid = np.array([np.linspace(-L2, L2, s + 1) for s in shape])

        field = generate_field(distrib, Pkgen(specIndex), shape, unit_length=L)
        field /= np.std(field)
        if expon:
            field = np.exp(log_offset + scale * field)
        return xgrid, field

    def recenterField(self, field):
        """Roll a periodic field so its mass-weighted center lies at the center of the box.

        The center of mass along each axis is the weighted mean direction of the
        1D mass profile, with pixel index mapped to angle, so structure that
        wraps across the periodic boundary is handled correctly.

        Parameters
        ----------
        field : ndarray
            Non-negative, periodic density field of any dimension.

        Returns
        -------
        ndarray
            `field` rolled by a whole number of cells along each axis so
            that its center of mass lies in the cell nearest
            ``shape // 2``. The values are unchanged, only shifted.
        """
        com = np.zeros(field.ndim)
        for ax, n in enumerate(field.shape):
            # Collapse to the 1D mass profile along this axis
            other = tuple(i for i in range(field.ndim) if i != ax)
            m = field.sum(axis=other)
            # Map pixel index -> angle, take the weighted mean direction
            theta = 2.0 * np.pi * np.arange(n) / n
            ang = np.arctan2(np.sum(m * np.sin(theta)), np.sum(m * np.cos(theta)))
            com[ax] = (ang % (2.0 * np.pi)) * n / (2.0 * np.pi)

        # Shift needed to bring the center of mass to the box center
        shift = np.rint(np.array(field.shape) // 2 - com).astype(int)
        return np.roll(field, shift, axis=tuple(range(field.ndim)))

    def makeCloudFBM(
        self,
        ndim=3,
        D=None,
        H=None,
        L=1.0,
        Ms=5.0,
        bturb=0.5,
        n0=1e2,
        min_dens=1.0,
        magBeta=1e6,
        nres=128,
        recenter=False,
        overSeed=None,
    ):
        """Generate a turbulent gas cloud as a log-normal fBm density field.

        The width of the log-normal density PDF is set by the turbulence,
        sigma^2 = ln(1 + bturb^2 Ms^2 magBeta / (1 + magBeta)) (e.g. Padoan
        & Nordlund 2011), and the density is n = n0 exp(sigma g) +
        `min_dens`, where g is a unit-variance fBm.

        Parameters
        ----------
        ndim : int, optional
            Number of spatial dimensions. Default is 3.
        D : float, optional
            Fractal dimension of the projected cloud (see :meth:`makeFBM`).
            Cannot be given together with `H`. If neither is given,
            defaults to 2.4.
        H : float, optional
            Hurst exponent of the log-density field (see :meth:`makeFBM`).
            Cannot be given together with `D`.
        L : float, optional
            Side length of the box; the grid spans [-L/2, L/2]. Default
            is 1.0.
        Ms : float, optional
            Sonic Mach number of the turbulence. Default is 5.0.
        bturb : float, optional
            Turbulent forcing parameter (1/3 solenoidal, 1 compressive).
            Default is 0.5.
        n0 : float, optional
            Median density of the log-normal part of the field. Default
            is 1e2.
        min_dens : float, optional
            Uniform density floor added to the field. Default is 1.0.
        magBeta : float, optional
            Plasma beta (thermal to magnetic pressure). Large values give
            the hydrodynamic limit. Default is 1e6.
        nres : int, optional
            Number of grid cells along each axis. Default is 128.
        recenter : bool, optional
            If True, roll the cloud so its center of mass lies at the
            center of the box (see :meth:`recenterField`). The floor is
            added after recentering so it does not dilute the weighting.
            Default is False.
        overSeed : int, optional
            Seed to use for this call instead of the object's `seed` (see
            :meth:`makeFBM`).

        Returns
        -------
        xgrid : ndarray
            Cell edges, of shape (ndim, nres + 1).
        cloud : ndarray
            Density field, of shape (nres,) * ndim.

        Raises
        ------
        ValueError
            If both `D` and `H` are given.
        """
        lnorm = np.log(n0)
        # How broad the n-PDF is
        sigma = np.sqrt(np.log(1.0 + bturb**2 * Ms**2 * (magBeta / (1.0 + magBeta))))

        xgrid, cloud = self.makeFBM(
            ndim=ndim,
            D=D,
            H=H,
            L=L,
            nres=nres,
            expon=True,
            scale=sigma,
            log_offset=lnorm,
            overSeed=overSeed,
        )
        if recenter:
            # Recenter before adding the floor so the uniform min_dens doesn't dilute the weighting
            cloud = self.recenterField(cloud)
        cloud += min_dens
        return xgrid, cloud

    def makeStellarCluster(
        self,
        nstar,
        ndim=3,
        D=None,
        H=None,
        L=1.0,
        Ms=None,
        bturb=None,
        magBeta=None,
        sigma=1,
        massSegregate=False,
        S=None,
        masses=None,
        recenter=False,
        nres=128,
        overSeed=None,
    ):
        """Sample protostar positions from a log-normal fBm density field.

        A log-fBm is generated with :meth:`makeFBM` and treated as a
        piecewise-constant probability density: each star is placed in a
        grid cell with probability proportional to the cell's density,
        then at a uniform random position within that cell. The positions
        can optionally be mass segregated with :meth:`segregate`.

        Parameters
        ----------
        nstar : int
            Number of stars to sample.
        ndim : int, optional
            Number of spatial dimensions. Default is 3.
        D : float, optional
            Fractal dimension of the projected density field (see
            :meth:`makeFBM`). Cannot be given together with `H`. If
            neither is given, defaults to 2.4.
        H : float, optional
            Hurst exponent of the log-density field (see :meth:`makeFBM`).
            Cannot be given together with `D`.
        L : float, optional
            Side length of the box; positions lie in [-L/2, L/2]. Default
            is 1.0.
        Ms : float, optional
            Sonic Mach number. If given, `sigma` is instead set from the
            turbulence as in :meth:`makeCloudFBM`, and `bturb` and
            `magBeta` must also be given.
        bturb : float, optional
            Turbulent forcing parameter. Required if `Ms` is given.
        magBeta : float, optional
            Plasma beta. Required if `Ms` is given.
        sigma : float, optional
            Standard deviation of the log density, controlling how
            strongly clustered the stars are. Ignored if `Ms` is given.
            Default is 1.
        massSegregate : bool, optional
            If True, mass segregate the positions using `S` and `masses`.
            Default is False.
        S : float, optional
            Degree of mass segregation, in [0, 1). 0 gives no segregation;
            values approaching 1 place the most massive stars in the most
            bound positions. Required if `massSegregate` is True.
        masses : array_like, optional
            Stellar masses, of length `nstar`. Required if `massSegregate`
            is True.
        recenter : bool, optional
            If True, roll the density field so its center of mass lies at
            the center of the box before sampling (see
            :meth:`recenterField`). Default is False.
        nres : int, optional
            Number of grid cells along each axis of the density field.
            For a cloud from :meth:`makeCloudFBM` and a cluster to share
            the same structure, they must use the same `nres`, seed,
            `ndim`, and `D` or `H`. Default is 128.
        overSeed : int, optional
            Seed to use for this call instead of the object's `seed` (see
            :meth:`makeFBM`).

        Returns
        -------
        tuple of ndarray
            One array of length `nstar` per dimension, so for ``ndim=3``
            it unpacks as ``x, y, z``. With mass segregation, star ``i``
            has mass ``masses[i]``.

        Raises
        ------
        ValueError
            If both `D` and `H` are given, if `Ms` is given without
            `bturb` and `magBeta`, if `S` is outside [0, 1), or if
            `massSegregate` is True without valid `S` and `masses`.
        """
        if Ms != None:
            if bturb == None or magBeta == None:
                raise ValueError(
                    "[OcotilloPMF Error] If using the physical turbulence for scaling, must give Ms, bturb and magBeta."
                )
            sigma = np.sqrt(
                np.log(1.0 + bturb**2 * Ms**2 * (magBeta / (1.0 + magBeta)))
            )

        if (S != None) and ((S < 0) or (S >= 1)):
            raise ValueError(
                "[OcotilloPMF error] When using mass segregation, the mass segregation parameter must be [0, 1)"
            )

        if massSegregate:
            if S == None or masses is None:
                raise ValueError(
                    "[OcotilloPMF error] When using mass segregation, must give both S and masses."
                )
            masses = np.asarray(masses)
            if masses.shape != (nstar,):
                raise ValueError(
                    "[OcotilloPMF error] masses must be a 1D array of length nstar."
                )

        # One generator for the whole call: the field, the star sampling and the
        # segregation continue the same stream, so a seed gives the same cluster
        rng = self._rng(overSeed)
        xgrid, fbm = self.makeFBM(
            ndim=ndim, D=D, H=H, L=L, nres=nres, scale=sigma, rng=rng
        )
        if recenter:
            fbm = self.recenterField(fbm)

        # Treat the log-fBm as a piecewise-constant PDF: each cell's probability
        # is proportional to its density
        pdf = fbm.ravel() / np.sum(fbm)

        # Draw which cell each star lands in, then convert to per-axis indices
        cells = rng.choice(pdf.size, size=nstar, p=pdf)
        idx = np.unravel_index(cells, fbm.shape)

        # Place each star uniformly within its cell (xgrid holds cell edges)
        coords = []
        for d in range(ndim):
            edges = xgrid[d]
            left = edges[idx[d]]
            width = edges[idx[d] + 1] - left
            coords.append(left + rng.uniform(size=nstar) * width)

        if massSegregate:
            # Soften on the grid scale, below which the positions carry no structure
            coords = self.segregate(coords, masses, S, soft=L / fbm.shape[0], rng=rng)

        # For ndim=3 this unpacks as x, y, z; star i has mass masses[i]
        return tuple(coords)

    def segregate(self, coords, masses, S, soft=1e-2, rng=None):
        """Mass segregate a set of positions following Baumgardt et al. (2008), as in McLuster.

        Positions are ranked from most to least bound by their equal-mass
        potential. Stars are then visited from heaviest to lightest, and each
        is swapped into the position at index j = (1 - u^(1-S)) * N_remaining
        of the positions not yet taken, with u ~ U[0, 1). S = 0 gives a random
        assignment (no segregation); S -> 1 puts the i-th most massive star at
        the i-th most bound position.

        Parameters
        ----------
        coords : list of ndarray
            One array of positions per dimension, each of length N.
        masses : ndarray
            Stellar masses, of length N.
        S : float
            Degree of mass segregation, in [0, 1).
        soft : float, optional
            Softening length for the potential, in the same units as
            `coords`. Default is 1e-2.
        rng : numpy.random.RandomState, optional
            Generator to draw from. If None, a new generator seeded with
            the object's `seed` is used (randomly seeded if `seed` is
            None).

        Returns
        -------
        list of ndarray
            `coords` reordered so that star ``i``, with mass
            ``masses[i]``, sits at ``(coords[0][i], coords[1][i], ...)``.
            The set of positions is unchanged.
        """
        pos = np.column_stack(coords)
        nstar = len(pos)

        # Softened equal-mass potential of each position, in chunks to bound memory
        phi = np.zeros(nstar)
        chunk = max(1, int(4e6) // nstar)
        for start in range(0, nstar, chunk):
            r = sdist.cdist(pos[start : start + chunk], pos)
            # Remove the self term, which contributes 1/soft
            phi[start : start + chunk] = (
                -np.sum(1.0 / np.sqrt(r**2 + soft**2), axis=1) + 1.0 / soft
            )

        free = list(np.argsort(phi))  # most bound first
        byMass = np.argsort(-masses, kind="stable")  # heaviest first
        if rng is None:
            rng = self._rng()
        u = rng.uniform(size=nstar)

        assign = np.empty(nstar, dtype=int)
        for k, i in enumerate(byMass):
            nrem = nstar - k
            j = min(int((1.0 - u[k] ** (1.0 - S)) * nrem), nrem - 1)
            assign[i] = free.pop(j)

        return [c[assign] for c in coords]
