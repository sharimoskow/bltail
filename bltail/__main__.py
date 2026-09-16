"""Command line.

  python -m bltail --expr "2 + 0.5*sin(2*pi*y1) + 0.7*sin(2*pi*(y1+y2))" --angles 0:360:2 --out d.csv
  python -m bltail --array a_samples.npy --angles 0,45,58.28
  python -m bltail --example inclusion --angles 0:360:5 --method sqrt
"""
import argparse, sys, time
import numpy as np
from . import tail, sine_coefficient, inclusion_coefficient

def parse_angles(s):
    if ":" in s:
        a, b, st = (float(v) for v in s.split(":")); return np.arange(a, b, st)
    return np.array([float(v) for v in s.split(",")])

def main(argv=None):
    p = argparse.ArgumentParser(description="boundary layer tail d(n) of a periodic coefficient a(y1,y2) on [0,1)^2")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--expr", help="formula in y1, y2 (numpy functions and pi available)")
    src.add_argument("--array", help=".npy file with samples a[i,j] = a(i/N, j/N)")
    src.add_argument("--example", choices=["sine", "inclusion"], help="the two coefficients of the paper")
    p.add_argument("--angles", default="0:360:2", help="'start:stop:step' or comma-separated list, degrees")
    p.add_argument("--K", type=int, default=8, help="Fourier modes |k|_inf <= K (default 8)")
    p.add_argument("--method", choices=["doubling", "sqrt"], default="doubling")
    p.add_argument("--out", help="write 'theta_deg,d' as CSV (or .npy if the name ends in .npy)")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    if a.expr: coef = a.expr
    elif a.array: coef = np.load(a.array)
    else: coef = sine_coefficient() if a.example == "sine" else inclusion_coefficient()
    angles = parse_angles(a.angles)
    t0 = time.time()
    d, cell = tail(coef, angles_deg=angles, K=a.K, method=a.method, verbose=not a.quiet, return_cell=True)
    if a.quiet:
        for th, v in zip(angles, d): print(f"{th:8.3f}  {v: .8e}")
    print(f"# a* = {cell.astar.tolist()}   ({time.time()-t0:.1f}s, {a.method}, K={a.K})", file=sys.stderr)
    if a.out:
        if a.out.endswith(".npy"): np.save(a.out, np.stack([angles, d], 1))
        else: np.savetxt(a.out, np.stack([angles, d], 1), delimiter=",", header="theta_deg,d", comments="")
        print("saved", a.out, file=sys.stderr)

if __name__ == "__main__":
    main()
