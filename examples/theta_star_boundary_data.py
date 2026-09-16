"""Boundary data of theta* on a smooth domain (an ellipse), for a user-supplied coefficient.

theta* solves -div(a* grad theta*) = 0 with theta* = d(n(x)) d_n u^0(x) on the boundary.
This script computes d(n(x)) along the boundary of an ellipse with any FEM-ready output:
points, outward normals, and d.  Run:  python examples/theta_star_boundary_data.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import bltail

a = "1 + 9*exp(-((y1-0.4)**2 + (y2-0.55)**2)/0.02)"      # a smooth bump of contrast 10 in the cell
s = np.linspace(0, 2*np.pi, 721)[:-1]                     # ellipse x = (2 cos s, sin s)
x = np.stack([2*np.cos(s), np.sin(s)], 1)
tangent = np.stack([-2*np.sin(s), np.cos(s)], 1)
normals = np.stack([tangent[:, 1], -tangent[:, 0]], 1)    # outward
angles, dcurve = bltail.tail_curve(a, step_deg=2.0, K=8, verbose=True)   # 180 directions, ~5 min
th = np.degrees(np.arctan2(normals[:, 1], normals[:, 0])) % 360
d = np.interp(th, np.append(angles, 360.0), np.append(dcurve, dcurve[0]))  # d(n(x)) at the 720 boundary points
# (equivalently: d = bltail.boundary_data(a, normals, step_deg=2.0, K=8), which builds the same table)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
g = np.linspace(0, 1, 200)
im = ax[0].imshow(bltail.as_coefficient(a).eval(g[:, None], g[None, :]).T, origin="lower", extent=[0, 1, 0, 1]); ax[0].set_title("a(y)"); plt.colorbar(im, ax=ax[0])
ax[1].plot(angles, dcurve); ax[1].set_xlabel("normal angle (deg)"); ax[1].set_ylabel("d(n)"); ax[1].set_title("tail d(n)"); ax[1].grid(alpha=.3)
sc = ax[2].scatter(x[:, 0], x[:, 1], c=d, s=8, cmap="viridis"); ax[2].set_aspect("equal"); ax[2].set_title("d(n(x)) on the ellipse"); plt.colorbar(sc, ax=ax[2])
plt.tight_layout(); plt.savefig("theta_star_boundary_data.png", dpi=120)
np.savetxt("ellipse_d.csv", np.column_stack([x, normals, d]), delimiter=",", header="x1,x2,n1,n2,d", comments="")
print("wrote theta_star_boundary_data.png and ellipse_d.csv;  d in [%.4e, %.4e]" % (d.min(), d.max()))
