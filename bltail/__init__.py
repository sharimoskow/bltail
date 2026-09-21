"""bltail — boundary layer tails in periodic homogenization via the commutator-Riccati cell problem.

Quick start:
    import bltail
    d = bltail.tail("2 + 0.5*sin(2*pi*y1) + 0.7*sin(2*pi*(y1+y2))", angles_deg=[0, 45, 58.28])
"""
from .coefficient import FourierCoefficient, TrigCoefficient, sine_coefficient, inclusion_coefficient
from .cell import Cell
from .tail import dtn_doubling, dtn_sqrt, density, sweep, far_field, tails
from .api import tail, tail_curve, boundary_data, as_coefficient
__version__ = "0.3.0"
__all__ = ["tail", "tail_curve", "boundary_data", "as_coefficient",
           "FourierCoefficient", "TrigCoefficient", "sine_coefficient", "inclusion_coefficient",
           "Cell", "dtn_doubling", "dtn_sqrt", "density", "sweep", "far_field", "tails"]
