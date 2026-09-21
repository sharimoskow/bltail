"""Cell problems in a Fourier basis on T^2 = [0,1)^2.

Conventions (same as the paper):
    cell correctors        -div( a (e_j - grad chi^j) ) = 0,   mean(chi^j) = 0
    homogenized matrix     a*_{ij} = < a (delta_ij - d_i chi^j) >
    normal corrector       chi_n = n_j chi^j
Fourier basis e_k(y) = exp(2 pi i k.y), |k|_inf <= K, with the *mean-normalised* inner product
< f, g > = int_{T^2} conj(f) g,  so that  < f, g > = sum_k conj(f_k) g_k.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve


class Cell:
    """Fourier-Galerkin discretisation of the unit cell for a coefficient with a ``.hat(k)`` method.

    Attributes
    ----------
    k : (M, 2) integer array of Fourier modes.
    A : (M, M) Hermitian Toeplitz matrix of multiplication by a.
    chi : (2, M) Fourier coefficients of the two cell correctors.
    astar : (2, 2) homogenized matrix.
    """

    def __init__(self, coef, K: int = 8):
        self.coef, self.K = coef, K
        ks = np.arange(-K, K + 1)
        K1, K2 = np.meshgrid(ks, ks, indexing="ij")
        self.k = np.stack([K1.ravel(), K2.ravel()], axis=1)
        self.M = len(self.k)
        self.zero = int(np.where((self.k[:, 0] == 0) & (self.k[:, 1] == 0))[0][0])
        # multiplication operator, A[m, m'] = a_hat[k_m - k_m']  (needs |dk|_inf <= 2K)
        cache = {}
        def ah(d):
            if d not in cache:
                cache[d] = coef.hat(d)
            return cache[d]
        A = np.empty((self.M, self.M), dtype=complex)
        for m, kk in enumerate(self.k):
            for mp, kp in enumerate(self.k):
                A[m, mp] = ah((int(kk[0] - kp[0]), int(kk[1] - kp[1])))
        self.A = 0.5 * (A + A.conj().T)
        # cell correctors:  sum_k' 4 pi^2 (k.k') a_{k-k'} chi_k' = -2 pi i k_j a_k
        Lcell = (4 * np.pi**2 * (self.k @ self.k.T)) * self.A
        chi = []
        for j in range(2):
            rhs = np.array([-2j * np.pi * kk[j] * ah((int(kk[0]), int(kk[1]))) for kk in self.k])
            Lc = Lcell.copy(); Lc[self.zero, :] = 0; Lc[self.zero, self.zero] = 1; rhs[self.zero] = 0
            chi.append(solve(Lc, rhs))
        self.chi = np.array(chi)
        self.e0 = np.zeros(self.M, dtype=complex); self.e0[self.zero] = 1.0
        self.astar = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                v = (self.e0 if i == j else 0) - 2j * np.pi * self.k[:, i] * self.chi[j]
                self.astar[i, j] = float(np.real(self.e0.conj() @ (self.A @ v)))

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _unit(n):
        n = np.asarray(n, dtype=float)
        if n.shape == ():                       # an angle in radians
            n = np.array([np.cos(n), np.sin(n)])
        return n / np.linalg.norm(n)

    def normal_data(self, n):
        """For a unit normal ``n`` (2-vector, or angle in radians) return
        ``chin, w, q, A_nn, T1`` with w_k = 2 pi k.n, q_k = 2 pi k.n_perp,
        A_nn = n.a* n and T1 = < a chi_n (1 - d_n chi_n) > (the 'toric term')."""
        n = self._unit(n); nperp = np.array([-n[1], n[0]])
        chin = n[0] * self.chi[0] + n[1] * self.chi[1]
        w = 2 * np.pi * (self.k @ n); q = 2 * np.pi * (self.k @ nperp)
        flux = self.A @ (self.e0 - 1j * w * chin)          # a (1 - d_n chi_n)
        Ann = float(np.real(self.e0.conj() @ flux)); T1 = float(np.real(chin.conj() @ flux))
        return chin, w, q, Ann, T1

    def tangential_corrector(self, n):
        """chi_tau = tau_j chi^j for the counterclockwise unit tangent tau = (-n_2, n_1)."""
        n = self._unit(n)
        return -n[1] * self.chi[0] + n[0] * self.chi[1]

    def project(self, f, ngrid: int = 256):
        """Fourier coefficient vector (on this cell's modes) of a periodic datum ``f``: a coefficient
        object with ``.hat``, a callable ``f(y1, y2)``, a string in y1, y2, or a 2D array of samples."""
        from .api import as_coefficient
        coef = as_coefficient(f, ngrid=ngrid)
        return np.array([coef.hat((int(kk[0]), int(kk[1]))) for kk in self.k], dtype=complex)

    def to_grid(self, coefs, N: int = 128):
        """Evaluate a Fourier vector on the N x N grid (i/N, j/N)."""
        F = np.zeros((N, N), dtype=complex)
        for m, kk in enumerate(self.k):
            F[kk[0] % N, kk[1] % N] = coefs[m]
        return np.real(np.fft.ifft2(F) * N * N)
