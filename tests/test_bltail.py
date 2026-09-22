import numpy as np
import bltail
import pytest
from bltail import TrigCoefficient, sine_coefficient, inclusion_coefficient, Cell, dtn_doubling, dtn_sqrt, density

def test_constant_coefficient():
    cell = Cell(TrigCoefficient({(0, 0): 3.0}), 4)
    assert np.allclose(cell.astar, 3 * np.eye(2))
    r = dtn_doubling(cell, 0.3)
    assert abs(r["d"]) < 1e-12
    # N = a |q| for constant a: check on a single mode
    chin, w, q, Ann, T1 = cell.normal_data(0.3)
    assert np.allclose(np.diag(r["N"]).real, 3 * np.abs(q), atol=1e-8)
    s = dtn_sqrt(cell, 0.3)
    assert np.allclose(s["N"], r["N"], atol=1e-8)

def test_sine_e1_reference_value():
    # phase-averaged strip solves (finite volume, Richardson-extrapolated) give 5.4233e-3 for
    # the exterior DtN form at n = e_1;  A_11 = 1.866788
    cell = Cell(sine_coefficient(), 8)
    r = dtn_doubling(cell, 0.0, orientation="exterior")
    assert abs(r["E"] - 5.42335e-3) < 2e-7
    assert abs(r["Ann"] - 1.866788) < 1e-5

def test_dt_independence():
    cell = Cell(sine_coefficient(), 6)
    e = [dtn_doubling(cell, 0.7, dt=dt)["E"] for dt in (0.05, 0.5, 2.0)]
    assert max(e) - min(e) < 1e-10 * max(e)

def test_riccati_residual():
    cell = Cell(sine_coefficient(), 6)
    chin, w, q, Ann, T1 = cell.normal_data(0.7)
    r = dtn_doubling(cell, 0.7, orientation="interior")
    N = r["N"]; A = cell.A; G = np.diag(1j * w)
    Lp = np.diag(q) @ A @ np.diag(q)
    res = N @ np.linalg.inv(A) @ N + G @ N - N @ G - Lp
    assert np.linalg.norm(res) < 1e-8 * np.linalg.norm(Lp)

def test_density_integrates_to_one_and_gives_d():
    cell = Cell(sine_coefficient(), 6)
    rho, r = density(cell, 0.4, N=64)
    assert abs(rho.mean() - 1.0) < 1e-10
    chin_grid = cell.to_grid(r["chin"], 64)
    assert abs((rho * chin_grid).mean() - r["d"]) < 1e-10

def test_inclusion_centrosymmetry_broken():
    cell = Cell(inclusion_coefficient(ngrid=256), 4)
    _, _, _, _, T1 = cell.normal_data(0.0)
    assert abs(T1) > 1e-5

def test_api_inputs_agree():
    import bltail
    expr = "2 + 0.5*sin(2*pi*y1) + 0.7*sin(2*pi*(y1+y2))"
    d_expr = bltail.tail(expr, angles_deg=[0.0, 45.0], K=6, check_K=False)
    g = np.arange(256)/256
    arr = 2 + 0.5*np.sin(2*np.pi*g[:, None]) + 0.7*np.sin(2*np.pi*(g[:, None]+g[None, :]))
    d_arr = bltail.tail(arr, angles_deg=[0.0, 45.0], K=6, check_K=False)
    d_obj = bltail.tail(sine_coefficient(), normals=[[1, 0], [1, 1]], K=6, check_K=False)
    assert np.allclose(d_expr, d_obj, rtol=1e-8)
    assert np.allclose(d_arr, d_obj, rtol=2e-3)        # bilinear samples of a
    assert abs(d_obj[0] - 2.905176e-3) < 1e-6

def test_boundary_data_interpolation():
    import bltail
    nn = np.array([[np.cos(0.3), np.sin(0.3)]])
    d_exact = bltail.boundary_data(sine_coefficient(), nn, K=6, check_K=False)
    d_interp = bltail.boundary_data(sine_coefficient(), nn, K=6, check_K=False, step_deg=2.0)
    assert abs(d_exact[0] - d_interp[0]) < 0.02*abs(d_exact[0])

def test_negative_coefficient_rejected():
    import bltail, pytest
    with pytest.raises(ValueError):
        bltail.tail("sin(2*pi*y1)", angles_deg=[0.0], K=4, check_K=False)


def test_far_field_and_tails():
    """The density represents the far field for every datum: linearity, normalization, and the
    two tails; a constant coefficient has zero tails."""
    cell = bltail.Cell(bltail.sine_coefficient(), 6)
    n = np.array([np.cos(0.9), np.sin(0.9)])
    chin = cell.normal_data(n)[0]
    d, dtau = bltail.tails(cell, n)
    assert abs(bltail.far_field(cell, n, chin) - d) < 1e-12
    assert abs(bltail.far_field(cell, n, "1 + 0*y1") - 1.0) < 1e-10
    assert abs(bltail.far_field(cell, n, chin + 2 * cell.e0) - (d + 2)) < 1e-10
    assert abs(bltail.far_field(cell, n, cell.tangential_corrector(n)) - dtau) < 1e-12
    c0 = bltail.Cell(bltail.TrigCoefficient({(0, 0): 1.0}), 4)
    assert bltail.tails(c0, n) == (0.0, 0.0)
    out = bltail.tail(bltail.sine_coefficient(), normals=[n], K=6, datum="tangential", check_K=False)
    assert abs(out[0] - dtau) < 1e-12


def test_laminate_along_layers():
    """Laminate a = a(y1), boundary along the layers (n = e2): chi_n = 0, rho = a/<a>, and the
    tangential tail is the a-weighted average of chi^1 (Moskow's thesis), in closed form."""
    rng = np.random.default_rng(3); N = 65536; y = (np.arange(N) + 0.5) / N
    coefs = [(rng.normal(), rng.uniform(0, 7)) for k in range(1, 6)]
    def afun(y1, y2):
        v = 3 + sum(c * np.sin(2 * np.pi * k * y1 + p) / k for k, (c, p) in enumerate(coefs, 1))
        return np.abs(v) + 0.5
    a = afun(y, 0); A = 1 / np.mean(1 / a); dchi = 1 - A / a
    chi = np.cumsum(dchi) / N - dchi / (2 * N); chi -= chi.mean()
    exact = np.mean(a * chi) / np.mean(a)
    cell = bltail.Cell(bltail.FourierCoefficient(afun, 1024), 12)
    assert abs(cell.astar[0, 0] - A) < 1e-4 and abs(cell.astar[1, 1] - np.mean(a)) < 1e-10
    d, dtau = bltail.tails(cell, [0.0, 1.0])
    assert abs(d) < 1e-12                       # chi^2 = 0 for a laminate along y1
    assert abs(dtau + exact) < 1e-4 * abs(exact)  # tau = (-1, 0): chi_tau = -chi^1
    rho, _ = bltail.density(cell, [0.0, 1.0], N=64)
    g = np.arange(64) / 64
    assert np.max(np.abs(rho - afun(g, 0)[:, None] / np.mean(a))) < 1e-3
