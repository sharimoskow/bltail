"""Independent check of the doubling at the golden angle:  python examples/check_far_field.py

The Fourier-truncated lifted problem  (d_t - iW) A (d_t - iW) V = Q A Q V  is solved exactly in the
depth t through its modes  e^{-mu t} phi,  (mu + iW) A (mu + iW) phi = Q A Q phi.  The bounded solution
with data chi_n is  c_0 * 1 + sum_{Re mu > 0} c_j e^{-mu_j t} phi_j,  and its far-field constant c_0
is compared with the doubling value of d(n).  This uses neither the doubling nor the DtN operator,
and checks that the density represents the far field (Theorem 3 of the paper).
"""
import numpy as np, scipy.linalg as sl
from bltail import sine_coefficient, inclusion_coefficient, Cell
from bltail.tail import dtn_doubling

def far_field_modal(cell, n):
    chin, w, q, Ann, T1 = cell.normal_data(n)
    A, M = cell.A, cell.M
    iW, Q = np.diag(1j * w), np.diag(q)
    M1 = A @ iW + iW @ A
    M0 = iW @ A @ iW - Q @ A @ Q
    Z, I = np.zeros((M, M)), np.eye(M)
    mu, V = sl.eig(np.block([[Z, I], [-M0, -M1]]), np.block([[I, Z], [Z, A]]))
    dec = np.where(mu.real > 1e-8)[0]
    assert len(dec) == M - 1, "expected M-1 decaying modes at an irrational normal"
    B = np.column_stack([cell.e0, V[:M, dec]])          # constant + decaying modes
    return np.linalg.solve(B, chin)[0].real, mu[dec].real.min()

theta = np.arctan((1 + 5 ** 0.5) / 2)                   # golden angle
n = np.array([np.cos(theta), np.sin(theta)])
for name, coef, K in [("sine", sine_coefficient(), 8), ("inclusion", inclusion_coefficient(), 12)]:
    cell = Cell(coef, K)
    d = dtn_doubling(cell, n)["d"]
    c0, rate = far_field_modal(cell, n)
    print(f"{name:10s} K={K:2d}  doubling d = {d:.12e}   modal far field = {c0:.12e}   "
          f"rel. diff {abs(c0 - d) / abs(d):.1e}   (slowest decay rate {rate:.3f})")
