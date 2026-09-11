"""The density rho(., n) on the cell:  python examples/plot_density.py sine 0"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bltail import sine_coefficient, inclusion_coefficient, Cell, density

ex = sys.argv[1] if len(sys.argv) > 1 else "sine"; deg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
coef = sine_coefficient() if ex == "sine" else inclusion_coefficient()
cell = Cell(coef, 8)
rho, r = density(cell, np.radians(deg), N=128)
print(f"d({deg} deg) = {r['d']:.6e},  rho in [{rho.min():.3f}, {rho.max():.3f}],  mean {rho.mean():.6f}")
g = np.arange(128) / 128
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(coef.eval(g[:, None], g[None, :]).T, origin="lower", extent=[0, 1, 0, 1]); ax[0].set_title("a(y)")
im = ax[1].imshow(rho.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma"); ax[1].set_title(f"rho(y, n), theta={deg} deg")
plt.colorbar(im, ax=ax[1]); plt.tight_layout(); plt.savefig(f"density_{ex}_{int(deg)}.png", dpi=130)
