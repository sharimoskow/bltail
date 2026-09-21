"""User-facing entry points.

    >>> import bltail, numpy as np
    >>> d = bltail.tail("2 + 0.5*sin(2*pi*y1) + 0.7*sin(2*pi*(y1+y2))", angles_deg=[0, 30, 58.28])
    >>> d = bltail.tail(lambda y1, y2: 1 + 9*((y1-0.5)**2 + (y2-0.5)**2 < 0.1), normals=[[1,0],[0.6,0.8]])
    >>> d = bltail.tail(a_samples_on_grid, angles_deg=np.arange(0, 360, 2))

The coefficient a(y) is Z^2-periodic on the unit cell [0,1)^2 and may be given as
  * a string in the variables y1, y2 (numpy functions and pi available),
  * a callable a(y1, y2) accepting numpy arrays,
  * a 2D array of samples a[i, j] = a(i/N, j/N)  (periodic bilinear interpolation is used),
  * an object with a .hat(k) method (see coefficient.py).
"""
from __future__ import annotations
import numpy as np
import warnings
from .coefficient import FourierCoefficient, TrigCoefficient
from .cell import Cell
from .tail import dtn_doubling, dtn_sqrt


# ---------------------------------------------------------------------------------------------
def as_coefficient(a, ngrid: int = 1024):
    """Turn any of the accepted descriptions of a(y) into a coefficient object."""
    if hasattr(a, "hat"):
        return a
    if isinstance(a, str):
        expr = a
        env = {k: getattr(np, k) for k in dir(np) if not k.startswith("_")}
        env["pi"] = np.pi
        def f(y1, y2):
            loc = dict(env); loc.update(y1=y1, y2=y2)
            return eval(expr, {"__builtins__": {}}, loc)   # noqa: S307  (user-supplied formula)
        return FourierCoefficient(f, ngrid)
    if callable(a):
        return FourierCoefficient(a, ngrid)
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 2:
        N1, N2 = arr.shape
        def f(y1, y2):
            x = np.asarray(y1) * N1; yy = np.asarray(y2) * N2
            i0 = np.floor(x).astype(int) % N1; j0 = np.floor(yy).astype(int) % N2
            tx = x - np.floor(x); ty = yy - np.floor(yy); i1 = (i0 + 1) % N1; j1 = (j0 + 1) % N2
            return ((1-tx)*(1-ty)*arr[i0, j0] + tx*(1-ty)*arr[i1, j0]
                    + (1-tx)*ty*arr[i0, j1] + tx*ty*arr[i1, j1])
        return FourierCoefficient(f, ngrid)
    raise TypeError("a must be a string, a callable a(y1,y2), a 2D array of samples, or a coefficient object")


def _check_positive(coef, n=256):
    g = np.arange(n) / n
    v = np.asarray(coef.eval(g[:, None], g[None, :]))
    if v.min() <= 0:
        raise ValueError(f"the coefficient must be positive; sampled minimum is {v.min():.3g}")
    return float(v.min()), float(v.max())


def _normals(angles_deg=None, normals=None):
    if (angles_deg is None) == (normals is None):
        raise ValueError("give exactly one of angles_deg or normals")
    if angles_deg is not None:
        th = np.radians(np.atleast_1d(np.asarray(angles_deg, dtype=float)))
        return np.stack([np.cos(th), np.sin(th)], axis=1)
    nn = np.atleast_2d(np.asarray(normals, dtype=float))
    return nn / np.linalg.norm(nn, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------
def tail(a, angles_deg=None, normals=None, K: int = 8, method: str = "doubling",
         check_K: bool = True, verbose: bool = False, return_cell: bool = False):
    """The boundary layer tail d(n) for a periodic coefficient a and a set of outward unit normals.

    Parameters
    ----------
    a          : the coefficient (string, callable, 2D array, or coefficient object).
    angles_deg : angles theta with n = (cos theta, sin theta), in degrees; or
    normals    : an (m, 2) array of (not necessarily unit) normal vectors.
    K          : Fourier truncation |k|_inf <= K (default 8; 12-16 for sharp coefficients).
    method     : 'doubling' (exact up to Fourier truncation) or 'sqrt' (square-root
                 approximation, ~20x cheaper, few-percent to 20% error).
    check_K    : estimate the truncation error by comparing K and K+4 at one normal.

    Returns
    -------
    d : array of tail values, one per normal (the paper's d(n), interior orientation,
        n the outward normal).  With return_cell=True also the Cell object (a*, correctors).
    """
    coef = as_coefficient(a)
    amin, amax = _check_positive(coef)
    nn = _normals(angles_deg, normals)
    cell = Cell(coef, K)
    solver = dtn_doubling if method == "doubling" else dtn_sqrt
    if check_K:
        cell2 = Cell(coef, K + 4)
        d1 = solver(cell, nn[0])["d"]; d2 = solver(cell2, nn[0])["d"]
        rel = abs(d1 - d2) / max(abs(d2), 1e-300)
        if verbose:
            print(f"a in [{amin:.3g}, {amax:.3g}], a* =\n{cell.astar.round(6)}\n"
                  f"truncation check at n={nn[0].round(4)}: K={K}: {d1:.6e}, K={K+4}: {d2:.6e}  (rel. change {rel:.1e})")
        if rel > 1e-3:
            warnings.warn(f"Fourier truncation K={K} may be too small for this coefficient "
                          f"(relative change {rel:.1e} when K -> {K+4}); increase K.")
    out = np.empty(len(nn))
    for i, n in enumerate(nn):
        out[i] = solver(cell, n)["d"]
        if verbose:
            print(f"  n = ({n[0]:+.4f}, {n[1]:+.4f})   d = {out[i]:+.6e}", flush=True)
    return (out, cell) if return_cell else out


def tail_curve(a, step_deg: float = 2.0, **kw):
    """d(n) on a uniform grid of angles 0, step, 2 step, ... < 360.  Returns (angles_deg, d)."""
    angles = np.arange(0.0, 360.0, step_deg)
    return angles, tail(a, angles_deg=angles, **kw)


def boundary_data(a, boundary_normals, dn_u0=None, **kw):
    """Boundary data of theta* on a smooth domain: d(n(x)) * d_n u^0(x) at boundary points.

    boundary_normals : (m, 2) outward normals at the boundary points.
    dn_u0            : normal derivative of the homogenized solution at those points
                       (if None, returns d(n(x)) alone).
    Each distinct normal is solved for once; for many points, pass step_deg=... in kw to
    compute d on a uniform angle grid and interpolate (linear in the angle; d is only
    Lipschitz at rational normals, so keep the step small, 1-2 degrees).
    """
    nn = _normals(normals=boundary_normals)
    step = kw.pop("step_deg", None)
    if step is None:
        d = tail(a, normals=nn, **kw)
    else:
        ang, dgrid = tail_curve(a, step_deg=step, **kw)
        th = np.degrees(np.arctan2(nn[:, 1], nn[:, 0])) % 360.0
        d = np.interp(th, np.append(ang, 360.0), np.append(dgrid, dgrid[0]))
    return d if dn_u0 is None else d * np.asarray(dn_u0, dtype=float)
