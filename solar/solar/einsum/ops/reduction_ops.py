"""Handlers for reduction operations.

This module provides einsum handlers for:
- sum, mean, prod
- max, min, amax, amin
- argmax, argmin
- logsumexp, norm
"""

import string
from typing import Any, List, Optional

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class ReductionHandler(EinsumOpHandler):
    """Handler for reduction operations."""
    
    supported_ops = [
        "sum", "mean", "prod",
        "max", "min", "amax", "amin",
        "argmax", "argmin",
        "logsumexp", "norm",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for reduction operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Get reduction dimensions
        dims = kwargs.get("dims")
        if dims is None:
            dims = kwargs.get("reduce_dims")
        
        keepdim = bool(kwargs.get("keepdim", False))
        
        # Normalize op name
        op_type = op_name.lower()
        if op_type in {"amax", "amin"}:
            op_type = op_type[1:]  # amax -> max, amin -> min
        
        return self._generate_reduction_einsum(
            input_shape, op_type, dims, keepdim
        )
    
    def _generate_reduction_einsum(
        self,
        shape: TensorShape,
        op_type: str = "sum",
        dims: Optional[List[int]] = None,
        keepdim: bool = False
    ) -> EinsumOp:
        """Generate einsum for reduction operations.
        
        Args:
            shape: Input tensor shape.
            op_type: Type of reduction (sum, mean, max, etc.).
            dims: Dimensions to reduce along.
            keepdim: Whether to keep reduced dimensions.
            
        Returns:
            EinsumOp for the reduction operation.
        """
        ndims = len(shape)
        input_labels = string.ascii_uppercase[:ndims]
        
        # Determine output labels based on reduction dims
        if dims is None:
            # Reduce all dimensions
            output_labels = ""
        else:
            # Keep non-reduced dimensions
            output_labels = ""
            for i, label in enumerate(input_labels):
                if i not in dims:
                    output_labels += label
        
        operands = [
            EinsumOperand("Input", list(input_labels), is_output=False),
            EinsumOperand("Output", list(output_labels) if output_labels else [], is_output=True),
        ]
        
        equation = f"{input_labels}->{output_labels}"
        
        # Map reduction op_type to appropriate reduction_op
        reduction_op_map = {
            "sum": "add",
            "mean": "add",  # Mean is sum then divide
            "prod": "mul",
            "max": "max",
            "min": "min",
            "argmax": "max",
            "argmin": "min",
            "logsumexp": "add",
            "norm": "add",
        }
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name=op_type,
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op=reduction_op_map.get(op_type, "add"),
        )


# Register handler with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(ReductionHandler)


__all__ = ["ReductionHandler"]

