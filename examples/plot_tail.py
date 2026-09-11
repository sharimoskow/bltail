"""Reproduce the d(n) figures of the paper:  python examples/plot_tail.py inclusion"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bltail import sine_coefficient, inclusion_coefficient, Cell, sweep

ex = sys.argv[1] if len(sys.argv) > 1 else "sine"
coef = sine_coefficient() if ex == "sine" else inclusion_coefficient()
cell = Cell(coef, 8)
angles = np.arange(0, 360, 2.0)
D = sweep(cell, angles, "doubling", verbose=True)
S = sweep(cell, angles, "sqrt")
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
ax[0].plot(angles, D[:, 1], "-", lw=2, label="Riccati doubling (exact)")
ax[0].plot(angles, S[:, 1], "--", lw=1.5, label="square-root approximation")
ax[0].set_ylabel("d(n)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(angles, 100 * (S[:, 1] - D[:, 1]) / D[:, 1], "k-"); ax[1].axhline(0, color="k", lw=.5)
ax[1].set_ylabel("rel. diff. (%)"); ax[1].set_xlabel("theta (deg)"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"tail_{ex}.png", dpi=130); print("saved", f"tail_{ex}.png")
