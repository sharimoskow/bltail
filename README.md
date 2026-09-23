# bltail — boundary layer tails in periodic homogenization

Computes the boundary layer tail `d(n)`, the half-space Dirichlet-to-Neumann operator `N_n`
and the boundary density `rho(., n)` of periodic homogenization **from a problem on the periodicity
cell**, for every unit normal `n` (rational or irrational), in two dimensions.

Companion code to

> S. Moskow, *A cell problem for the boundary density and approximation of boundary correctors
> in homogenization theory* (2026), preprint.

Repository: https://github.com/sharimoskow/bltail

If you use this code, please cite the paper (a `CITATION.cff` file is included).

## What it computes

For `-div(a(x/eps) grad u) = f` with `u = 0` on the boundary, the boundary corrector converges to
`theta*` with boundary data `d(n(x)) d_n u^0(x)`.  The tail `d(n)` is given by

```
A_nn d(n) = < a chi_n (1 - d_n chi_n) >  +  < chi_n , N_n chi_n >
```

where `chi_n = n_j chi^j` is the normal cell corrector and `N_n` is the Dirichlet-to-Neumann (DtN)
operator of the lifted half-space problem (Gérard-Varet–Masmoudi, in the form in which the cell
coordinate moves with the coefficient).  The paper shows that `N_n` is the symmetric non-negative
solution on the torus of the **commutator–Riccati equation**

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
  iterated to its fixed point.  The fixed point is **independent of `dt`** (a proposition in the paper: the Crank–Nicolson slab reproduces each decaying mode and its flux exactly) and is the discrete
  solution of the commutator–Riccati equation, so the only approximation is the Fourier
  truncation.  About a dozen doublings, each one Hermitian solve of size `(2K+1)^2`; a few seconds
  per direction at `K = 8`.  Robust at rational normals.
* **`dtn_sqrt`** — the *square-root approximation* `N0 = a sqrt(a^{-1} L_perp)`, obtained by
  dropping the commutator: one generalized eigenproblem, roughly 20x cheaper.  Exact through second
  order in the contrast of `a` but not beyond; errors of a few percent for mildly varying
  coefficients and 10–20 % for high-contrast inclusions.  Useful as a preview or as a seed for the
  doubling.

Newton iteration on the Riccati equation started from the square root converges quadratically but
each step is a Lyapunov equation whose operator becomes singular at rational normals (and
ill-conditioned near them: the small divisors); it is not provided.

## Install

```
pip install -e .            # numpy, scipy;  matplotlib for the examples
python -m pytest tests      # about two minutes
```

## Quick start: give it `a`, get `d(n)`

```python
import numpy as np, bltail

# the coefficient: a formula in y1, y2 on the unit cell [0,1)^2 ...
d = bltail.tail("2 + 0.5*sin(2*pi*y1) + 0.7*sin(2*pi*(y1+y2))", angles_deg=[0, 45, 58.28])

# ... or any Python function of numpy arrays ...
d = bltail.tail(lambda y1, y2: 1 + 9*np.exp(-((y1-0.4)**2 + (y2-0.55)**2)/0.02), normals=[[1, 0], [0.6, 0.8]])

# ... or samples on a grid, a[i, j] = a(i/N, j/N)
d = bltail.tail(a_samples, angles_deg=np.arange(0, 360, 2))
```

`tail` returns the boundary layer tail `d(n)` of the paper — interior orientation, `n` the
outward normal — one value per normal.  It builds the Fourier–Galerkin cell (`K = 8` modes per
direction by default), solves the commutator–Riccati cell problem by Riccati doubling for each
normal, and checks the truncation by recomputing one normal at `K + 4` (a warning is issued if the
value moves by more than `1e-3`; use `K=12` or `16` for sharp coefficients).  `method="sqrt"` gives
the square-root approximation instead, about 20x faster and a few percent to 20% off.

For the boundary data of the limit corrector `theta*` on a smooth domain, `d(n(x)) d_n u^0(x)`:

```python
d = bltail.boundary_data(a, outward_normals_at_boundary_points, dn_u0=dn_u0_values, step_deg=2.0)
```

which tabulates `d` on a 2-degree grid of angles and interpolates (the tabulated function `d(n)` is continuous but has kinks at the rational
normals of small period, so keep the step small); omit `step_deg` to solve every normal exactly.
`examples/theta_star_boundary_data.py` does this for an ellipse and writes the boundary points,
normals and `d` to a CSV ready for a finite element code.

### Other boundary data

The density `rho(., n)` represents the far field for *every* periodic
datum, so the same cell problem gives the tail for any datum: `bltail.far_field(cell, n, f)`
returns `int rho(., n) f`, and `bltail.tail(a, ..., datum=f)` does it for a list of normals.
The case that arises in practice is nonhomogeneous Dirichlet data `u^0 = g` on the boundary:
the cell-corrector term then has a tangential part, and the limit corrector has boundary data

```
theta* = d(n) d_n u^0 + d_tau(n) d_tau u^0,     d_tau(n) = int rho(., n) chi_tau,  tau = (-n_2, n_1),
```

with `chi_tau = tau_j chi^j` the tangential cell corrector.  `bltail.tails(cell, n)` returns
`(d, d_tau)`, `bltail.tail(a, ..., datum="tangential")` returns `d_tau`, and
`bltail.boundary_data(a, normals, dn_u0=..., dtau_u0=...)` assembles both terms.
(Neumann or Robin conditions lead to a different half-space problem and are not covered.)

Command line:

```
python -m bltail --expr "2 + 0.5*sin(2*pi*y1) + 0.7*sin(2*pi*(y1+y2))" --angles 0:360:2 --out d.csv
python -m bltail --array a_samples.npy --angles 0,45,58.28 --K 12
python -m bltail --example inclusion --angles 0:360:5 --method sqrt
```

Lower-level objects — `Cell` (correctors, `a*`, the Toeplitz matrix of `a`), `dtn_doubling`
(the DtN operator `N_n` itself), `dtn_sqrt`, `density` (the function `rho(., n)` on the cell),
`sweep` — are available for finer control; `examples/plot_tail.py` and
`examples/plot_density.py` reproduce the figures of the paper.

## Conventions

* Cell correctors `-div(a (e_j - grad chi^j)) = 0`, `a*_{ij} = < a (delta_ij - d_i chi^j) >`.
* `n` is the **outward** normal; the half-space is the interior side `{y.n < 0}`
  (`orientation="exterior"` gives the other side).  `d(n)` and `d(-n)` differ in general.
* Fourier coefficients are mean-normalised: `a(y) = sum_k a_hat[k] exp(2 pi i k.y)`.
* Piecewise-constant coefficients work but converge only like `O(1/K)`; smoothing the interfaces
  over a width of a few percent of the period restores fast convergence (see the paper).
* **Rational normals.**  At a rational normal the lifted problem contains all phases of the
  boundary line relative to the cell, and the value returned is the *phase average* of the strip
  tails of Moskow–Vogelius; the limit of the corrector along a particular sequence `eps -> 0`
  depends on the phase selected by the sequence.  See the next point.
* **Smooth domains only.**  At a rational normal the lifted problem contains all phases of the
  boundary line relative to the cell, and `d(n)` returned here is the *phase average*.  That is the
  right value at a rational normal of a smooth boundary (an isolated direction of a continuous
  family, of measure zero), but not on a *flat side* with rational normal: there the limit depends on where the
  side cuts the cell (the phase `s`) and on the subsequence in `eps`, and is the phase-dependent
  strip tail `d(n, s)` of Moskow–Vogelius, which requires a strip solve at that phase (not
  provided).  A flat side with an *irrational* normal is fine in the limit (no phase, no
  subsequence, the tail is `d(n)`), but at finite `eps` it behaves that way only if the side is
  long enough for the boundary data to equidistribute: the mode `k` of `chi_n` oscillates along the
  side with wavelength `eps / |k . n_perp|`, so one needs `L |k . n_perp| >> eps` for the modes that
  carry `chi_n`; near a low-period rational direction this fails and the side is effectively
  rational with a phase.  Corners are not covered at all.

## Validation (see `tests/` and the paper)

* Constant coefficient: `d = 0`, `N = a |grad_perp|`.
* Example 1 at `n = e_1`: the DtN form agrees with phase-averaged finite-difference strip solves
  (`5.4233e-3` extrapolated vs `5.42335e-3`).
* Example 2 at `n = e_1`: `2.04394e-2` (finite-volume strips) vs `2.04400e-2`.
* The commutator–Riccati residual of the computed `N_n` is at round-off level; the density
  integrates to one and reproduces `d(n)`.

## Use of AI

This package was developed with the help of an AI assistant (Claude, Anthropic), as was the
accompanying paper; the author directed the work and verified the results.

## License

MIT.
