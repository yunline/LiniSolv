import sys

if sys.version_info<(3,11):
    raise RuntimeError("LiniSolv requires Python version >= 3.11")

del sys # clean the namespace

from .linsolv import *
from .parser import parse_circuit

__all__ = [
    "Component",
    "Resistor",
    "Source",
    "LinearCircuitSolver",
    "parse_circuit",
    "LinisolvError",
    "CircuitError",
    "CircuitTopologyError",
    "CircuitSolverError",
]