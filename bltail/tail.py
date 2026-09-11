"""The boundary layer tail d(n), the DtN operator N_n and the density rho(.,n).

Theory (see the paper):  with G = n.grad and L_perp = -div_perp(a grad_perp) on T^2, the DtN
operator of the lifted half-space problem (interior side, n the outward normal) is the
symmetric non-negative solution of the commutator-Riccati equation

        N a^{-1} N + [G, N] = L_perp,

and     A_nn d(n) = T1(n) + < chi_n, N chi_n >,        A_nn rho = a (1 - d_n chi_n) + N chi_n.

Methods
-------
``dtn_doubling``  (recommended)  Riccati doubling: discretise the lifted problem in the depth
    variable t, eliminate slabs pairwise (depth 2^k dt), iterate the free-end DtN operator to its
    fixed point.  The fixed point is *independent of dt* and equals the discrete solution of the
    commutator-Riccati equation; the only approximation is the Fourier truncation.
``dtn_sqrt``  Frozen-coefficient approximation  N0 = a sqrt(a^{-1} L_perp)  (one generalized
    eigenproblem, ~20x cheaper, exact only when a is constant along n; errors of a few % for
    mildly varying a, 10-20 % for high-contrast inclusions).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve, eigh
from .cell import Cell


def _slab(cell: Cell, w, q, sigma: int, dt: float):
    A = cell.A
    Rp = np.diag(1 / dt + sigma * 0.5j * w)
    Rm = np.diag(-1 / dt + sigma * 0.5j * w)
    Qh = np.diag(0.5j * q)
    QAQ = Qh.conj().T @ A @ Qh
    P = dt * (Rm.conj().T @ A @ Rm + QAQ)      # top-top
    R = dt * (Rp.conj().T @ A @ Rp + QAQ)      # bottom-bottom
    Q = dt * (Rm.conj().T @ A @ Rp + QAQ)      # top-bottom
    return P, Q, R


def dtn_doubling(cell: Cell, n, orientation: str = "interior", dt: float = 1.0,
                 tol: float = 1e-11, max_doublings: int = 24, seed=None):
    """Riccati doubling.  Returns a dict with keys
    ``E`` (= <chi_n, N chi_n>), ``T1``, ``Ann``, ``d`` (= (T1+E)/Ann), ``N`` (the M x M DtN
    matrix in the Fourier basis), ``doublings``, ``history``.

    ``orientation``: 'interior' (half-space {y.n < 0}, n outward: the paper's d(n)) or 'exterior'.
    ``seed``: optional far-end DtN matrix (e.g. the square root) — improves early iterates only.
    """
    sigma = -1 if orientation == "interior" else +1
    chin, w, q, Ann, T1 = cell.normal_data(n)
    P, Q, R = _slab(cell, w, q, sigma, dt)
    M = cell.M
    far = np.zeros((M, M), dtype=complex) if seed is None else seed
    hist = []
    for it in range(max_doublings):
        Lam = P - Q @ solve(R + far, Q.conj().T, assume_a="her")
        val = float(np.real(chin.conj() @ (Lam @ chin)))
        hist.append(val)
        if it > 1 and abs(hist[-1] - hist[-2]) < tol * abs(val):
            break
        X = solve(R + P, np.concatenate([Q.conj().T, Q], axis=1), assume_a="her")
        X1, X2 = X[:, :M], X[:, M:]
        P, R, Q = P - Q @ X1, R - Q.conj().T @ X2, -Q @ X2
    Lam = 0.5 * (Lam + Lam.conj().T)
    return dict(E=val, T1=T1, Ann=Ann, d=(T1 + val) / Ann, N=Lam, doublings=it + 1,
                history=np.array(hist), chin=chin)


def dtn_sqrt(cell: Cell, n):
    """Square-root (frozen-coefficient) approximation.  Same keys as ``dtn_doubling`` plus the
    generalized eigenpairs ``lam``, ``F`` of  L_perp f = lam a f."""
    chin, w, q, Ann, T1 = cell.normal_data(n)
    A = cell.A
    Lp = np.diag(q) @ A @ np.diag(q); Lp = 0.5 * (Lp + Lp.conj().T)
    lam, F = eigh(Lp, A); lam = np.maximum(lam, 0.0)
    N0 = (A @ F) @ (np.sqrt(lam)[:, None] * (F.conj().T @ A))
    N0 = 0.5 * (N0 + N0.conj().T)
    c = F.conj().T @ (A @ chin)
    E = float(np.sum(np.sqrt(lam) * np.abs(c) ** 2))
    return dict(E=E, T1=T1, Ann=Ann, d=(T1 + E) / Ann, N=N0, lam=lam, F=F, chin=chin)


def tail(cell: Cell, n, method: str = "doubling", **kw) -> float:
    """The boundary layer tail d(n) (interior orientation, n outward)."""
    if method == "doubling":
        return dtn_doubling(cell, n, **kw)["d"]
    if method == "sqrt":
        return dtn_sqrt(cell, n)["d"]
    raise ValueError(method)


def density(cell: Cell, n, N: int = 128, method: str = "doubling", **kw):
    """The boundary density rho(., n) on an N x N grid of the cell,
    A_nn rho = a (1 - d_n chi_n) + N_n chi_n,  with int rho = 1 and d(n) = int rho chi_n."""
    r = dtn_doubling(cell, n, **kw) if method == "doubling" else dtn_sqrt(cell, n)
    chin, w, q, Ann, T1 = cell.normal_data(n)
    rho_hat = (cell.A @ (cell.e0 - 1j * w * chin) + r["N"] @ chin) / Ann
    return cell.to_grid(rho_hat, N), r


def sweep(cell: Cell, angles_deg, method: str = "doubling", verbose: bool = False, **kw):
    """d(n) for n = (cos theta, sin theta) over a list of angles (degrees).  Returns an array
    with columns [theta_deg, d, E, T1, A_nn]."""
    out = []
    for th in angles_deg:
        r = dtn_doubling(cell, np.radians(th), **kw) if method == "doubling" else dtn_sqrt(cell, np.radians(th))
        out.append([th, r["d"], r["E"], r["T1"], r["Ann"]])
        if verbose:
            print(f"theta={th:7.2f}  d={r['d']:+.6e}", flush=True)
    return np.array(out)
