# bltail — boundary layer tails in periodic homogenization

Computes the boundary layer tail `d(n)`, the half-space Dirichlet-to-Neumann operator `N_n`
and the boundary density `rho(., n)` of periodic homogenization **from a problem on the periodicity
cell**, for every unit normal `n` (rational or irrational), in two dimensions.

Companion code to

> S. Moskow, *A cell problem for the boundary density and approximation of boundary corrector
> limits in periodic homogenization* (2026).

## What it computes

For `-div(a(x/eps) grad u) = f` with `u = 0` on the boundary, the boundary corrector converges to
`theta*` with boundary data `d(n(x)) d_n u^0(x)`.  The tail `d(n)` is given by

```
A_nn d(n) = < a chi_n (1 - d_n chi_n) >  +  < chi_n , N_n chi_n >
```

where `chi_n = n_j chi^j` is the normal cell corrector and `N_n` is the DtN operator of the lifted
half-space problem (Gérard-Varet–Masmoudi).  The paper shows that `N_n` is the symmetric
non-negative solution on the torus of the **commutator–Riccati equation**

```
N a^{-1} N + [ n.grad , N ] = L_perp(n),        L_perp = -div_perp( a grad_perp )
```

and that the density `A_nn rho = a (1 - d_n chi_n) + N_n chi_n` represents the far-field
functional for every boundary datum and satisfies `N_n (rho / a) + d_n rho = 0`.

## Method

Everything is done in a Fourier basis on the cell (`|k|_inf <= K`; multiplication by `a` is a
Toeplitz matrix).  Two solvers are provided:

* **`dtn_doubling`** (recommended) — *Riccati doubling*.  The lifted problem is discretised in the
  depth variable, slabs are eliminated pairwise (depth `2^k dt`), and the free-end DtN operator is
  iterated to its fixed point.  The fixed point is **independent of `dt`** and is the discrete
  solution of the commutator–Riccati equation, so the only approximation is the Fourier
  truncation.  About a dozen doublings, each one Hermitian solve of size `(2K+1)^2`; a few seconds
  per direction at `K = 8`.  Robust at rational normals.
* **`dtn_sqrt`** — the frozen-coefficient approximation `N0 = a sqrt(a^{-1} L_perp)`, one
  generalized eigenproblem, roughly 20x cheaper.  Exact through second order in the contrast of
  `a`; errors of a few percent for mildly varying coefficients and 10–20 % for high-contrast
  inclusions.  Useful as a preview or as a seed for the doubling.

Newton iteration on the Riccati equation started from the square root converges quadratically but
each step is a Lyapunov equation whose operator becomes singular at rational normals (and
ill-conditioned near them: the small divisors); it is not provided.

## Install

```
pip install -e .            # numpy, scipy;  matplotlib for the examples
python -m pytest tests      # a few seconds
```

## Usage

```python
import numpy as np
from bltail import sine_coefficient, inclusion_coefficient, FourierCoefficient, Cell, \
                   dtn_doubling, dtn_sqrt, tail, density, sweep

cell = Cell(inclusion_coefficient(), K=8)     # cell correctors, a*, Toeplitz matrix of a
print(cell.astar)

r = dtn_doubling(cell, np.radians(58.28))     # n = (cos, sin) of the golden angle; n may also be a 2-vector
print(r["d"], r["E"], r["T1"], r["Ann"])      # tail, DtN form, toric term, n.a* n
print(dtn_sqrt(cell, np.radians(58.28))["d"]) # square-root approximation

rho, r = density(cell, np.radians(30), N=128) # the density on a 128 x 128 grid of the cell
table = sweep(cell, np.arange(0, 360, 2.0))   # columns: theta_deg, d, E, T1, A_nn

# your own coefficient (any smooth Z^2-periodic function of (y1, y2) in [0,1)^2):
mycoef = FourierCoefficient(lambda y1, y2: 1.5 + np.cos(2*np.pi*y1)*np.sin(2*np.pi*y2)**2)
```

Command line:

```
python -m bltail --example inclusion --K 8 --step 2 --method both --out d_inclusion.npy
```

`examples/plot_tail.py {sine|inclusion}` reproduces the `d(n)` figures of the paper;
`examples/plot_density.py {sine|inclusion} <angle>` plots the density.

## Conventions

* Cell correctors `-div(a (e_j - grad chi^j)) = 0`, `a*_{ij} = < a (delta_ij - d_i chi^j) >`.
* `n` is the **outward** normal; the half-space is the interior side `{y.n < 0}`
  (`orientation="exterior"` gives the other side).  `d(n)` and `d(-n)` differ in general.
* Fourier coefficients are mean-normalised: `a(y) = sum_k a_hat[k] exp(2 pi i k.y)`.
* Piecewise-constant coefficients work but converge only like `O(1/K)`; smoothing the interfaces
  over a width of a few percent of the period restores fast convergence (see the paper).

## Validation (see `tests/` and the paper)

* Constant coefficient: `d = 0`, `N = a |grad_perp|`.
* Example 1 at `n = e_1`: the DtN form agrees with phase-averaged finite-difference strip solves
  (`5.4233e-3` extrapolated vs `5.42335e-3`).
* Example 2 at `n = e_1`: `2.04394e-2` (finite-volume strips) vs `2.04400e-2`.
* The commutator–Riccati residual of the computed `N_n` is at round-off level; the density
  integrates to one and reproduces `d(n)`.

## License

MIT.
