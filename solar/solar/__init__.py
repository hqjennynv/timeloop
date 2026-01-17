"""Solar: PyTorch Model Einsum Toolkit.

This package provides tools for extracting PyTorch model graphs
and converting them to einsum representations.

Package structure:
- solar.einsum: Einsum conversion (ops, converters, analyzer)
- solar.graph: PyTorch graph processing
"""

__version__ = "1.2.0"

# Einsum conversion and analysis
from solar.einsum import (
    EinsumAnalyzer,
    PyTorchToEinsum,
    PyTorchEinsumConverter,  # Backward compatibility
    BenchmarkEinsumConverter,
)

# Graph processing
from solar.graph import PyTorchProcessor, TorchviewProcessor

# Core types
from solar.einsum.ops import EinsumOp, EinsumOperand, FFNOp, FFNOperand

__all__ = [
    # Einsum
    "EinsumAnalyzer",
    "PyTorchToEinsum",
    "PyTorchEinsumConverter",  # Backward compatibility
    "BenchmarkEinsumConverter",
    # Graph processing
    "PyTorchProcessor",
    "TorchviewProcessor",
    # Types
    "EinsumOp",
    "EinsumOperand",
    "FFNOp",
    "FFNOperand",
]
