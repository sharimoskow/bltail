"""Periodic coefficients a(y) on the unit cell [0,1)^2, represented by their Fourier coefficients.

All coefficients are *mean-normalised*:  a(y) = sum_k  a_hat[k] exp(2 pi i k.y).
"""
from __future__ import annotations
import numpy as np


class FourierCoefficient:
    """A real, Z^2-periodic coefficient given by a callable, with Fourier coefficients computed
    once by FFT on an ``ngrid x ngrid`` grid.  ``ngrid`` must resolve the coefficient.

    Parameters
    ----------
    func : callable ``func(y1, y2) -> array`` accepting numpy arrays in [0,1).
    ngrid : FFT resolution (default 1024).
    """

    def __init__(self, func, ngrid: int = 1024):
        self.func = func
        g = np.arange(ngrid) / ngrid
        Y1, Y2 = np.meshgrid(g, g, indexing="ij")
        self._F = np.fft.fft2(np.asarray(func(Y1, Y2), dtype=float)) / ngrid**2
        self.ngrid = ngrid

    def eval(self, y1, y2):
        y1 = np.asarray(y1, dtype=float) % 1.0
        y2 = np.asarray(y2, dtype=float) % 1.0
        return self.func(y1, y2)

    def hat(self, k) -> complex:
        return complex(self._F[k[0] % self.ngrid, k[1] % self.ngrid])


class TrigCoefficient:
    """A trigonometric polynomial given explicitly by a dict ``{(k1, k2): a_hat}``.
    The dict must be Hermitian-symmetric (``a_hat[-k] = conj(a_hat[k])``) for a real coefficient.
    """

    def __init__(self, modes: dict):
        self.modes = {tuple(k): complex(v) for k, v in modes.items()}

    def hat(self, k) -> complex:
        return self.modes.get((int(k[0]), int(k[1])), 0.0)

    def eval(self, y1, y2):
        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        out = np.zeros(np.broadcast(y1, y2).shape, dtype=complex)
        for (k1, k2), v in self.modes.items():
            out += v * np.exp(2j * np.pi * (k1 * y1 + k2 * y2))
        return out.real


# ---------------------------------------------------------------------------------------------
# The two coefficients used in the paper
# ---------------------------------------------------------------------------------------------

def sine_coefficient() -> TrigCoefficient:
    """Example 1 of the paper:  a(y) = 2 + 0.5 sin(2 pi y1) + 0.7 sin(2 pi (y1 + y2))."""
    return TrigCoefficient({(0, 0): 2.0, (1, 0): -0.25j, (-1, 0): 0.25j,
                            (1, 1): -0.35j, (-1, -1): 0.35j})


def _periodic_window(y, x0, x1, delta):
    w = np.zeros_like(y, dtype=float)
    for m in (-1, 0, 1):
        w += 0.5 * (np.tanh((y + m - x0) / delta) - np.tanh((y + m - x1) / delta))
    return w


def inclusion_coefficient(a0=1.0, a1=10.0, delta=0.06,
                          rects=((0.15, 0.5, 0.15, 0.5), (0.6, 0.85, 0.6, 0.8)),
                          ngrid=1024) -> FourierCoefficient:
    """Example 2 of the paper: smoothed inclusions of value ``a1`` in a background ``a0``.
    ``rects`` are (x0, x1, y0, y1) boxes; edges are tanh profiles of width ``delta``,
    periodised exactly.  Not centrosymmetric for the default boxes."""

    def f(y1, y2):
        s = np.zeros(np.broadcast(y1, y2).shape)
        for (x0, x1, yy0, yy1) in rects:
            s = s + _periodic_window(np.asarray(y1, float), x0, x1, delta) * \
                    _periodic_window(np.asarray(y2, float), yy0, yy1, delta)
        return a0 + (a1 - a0) * s

    return FourierCoefficient(f, ngrid)
