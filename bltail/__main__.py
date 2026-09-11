"""Command line:  python -m bltail --example inclusion --K 8 --step 2 --out d_inclusion.npy"""
import argparse, sys, time
import numpy as np
from . import sine_coefficient, inclusion_coefficient, Cell, sweep

def main(argv=None):
    p = argparse.ArgumentParser(description="boundary layer tail d(n) over the circle of normals")
    p.add_argument("--example", choices=["sine", "inclusion"], default="sine")
    p.add_argument("--K", type=int, default=8, help="Fourier modes |k|_inf <= K")
    p.add_argument("--step", type=float, default=2.0, help="angle step in degrees")
    p.add_argument("--method", choices=["doubling", "sqrt", "both"], default="both")
    p.add_argument("--out", default=None, help="save the table as .npy")
    a = p.parse_args(argv)
    coef = sine_coefficient() if a.example == "sine" else inclusion_coefficient()
    t0 = time.time(); cell = Cell(coef, a.K)
    print(f"cell: K={a.K}, M={cell.M}, a* =\n{cell.astar.round(6)}   ({time.time()-t0:.1f}s)")
    angles = np.arange(0.0, 360.0, a.step)
    cols = [angles]
    if a.method in ("doubling", "both"):
        t0 = time.time(); r = sweep(cell, angles, "doubling"); cols.append(r[:, 1]); print(f"doubling: {time.time()-t0:.1f}s")
    if a.method in ("sqrt", "both"):
        t0 = time.time(); s = sweep(cell, angles, "sqrt"); cols.append(s[:, 1]); print(f"sqrt:     {time.time()-t0:.1f}s")
    tab = np.stack(cols, axis=1)
    hdr = "theta_deg " + {"doubling": "d_doubling", "sqrt": "d_sqrt", "both": "d_doubling d_sqrt"}[a.method]
    print(hdr)
    for row in tab:
        print("  ".join(f"{v: .6e}" if i else f"{v:6.1f}" for i, v in enumerate(row)))
    if a.out:
        np.save(a.out, tab); print("saved", a.out)

if __name__ == "__main__":
    main()
