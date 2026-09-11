import numpy as np
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
