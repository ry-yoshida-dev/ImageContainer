"""Structural typing protocols for array-backed image containers."""

from .geometry import SupportsArrayGeometry
from .hash import SupportsArrayHash
from .process import SupportsArrayProcess

__all__ = ["SupportsArrayGeometry", "SupportsArrayHash", "SupportsArrayProcess"]
