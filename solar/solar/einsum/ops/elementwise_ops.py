"""Handlers for elementwise operations.

This module provides einsum handlers for:
- Unary elementwise: relu, sigmoid, tanh, gelu, softmax, abs, exp, log, etc.
- Binary elementwise: add, sub, mul, div
"""

import string
from typing import Any, List

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class UnaryElementwiseHandler(EinsumOpHandler):
    """Handler for unary elementwise operations."""
    
    supported_ops = [
        "relu", "sigmoid", "tanh", "gelu", "selu", "elu", "mish",
        "softmax", "log_softmax", "softplus", "hardswish", "hardsigmoid",
        "abs", "neg", "exp", "log", "sqrt", "rsqrt", "sin", "cos",
        "clamp", "clamp_", "relu_",
        "dropout", "dropout_",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for unary elementwise operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        return self._generate_elementwise_einsum(input_shape, op_name)
    
    def _generate_elementwise_einsum(
        self,
        shape: TensorShape,
        op_type: str = "elementwise"
    ) -> EinsumOp:
        """Generate einsum for unary elementwise operations.
        
        Args:
            shape: Input tensor shape.
            op_type: Type of elementwise operation.
            
        Returns:
            EinsumOp for the elementwise operation.
        """
        dims = len(shape)
        labels = string.ascii_uppercase[:dims]
        
        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Output", list(labels), is_output=True),
        ]
        
        equation = f"{labels}->{labels}"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name=op_type,
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op="none",
        )


class BinaryElementwiseHandler(EinsumOpHandler):
    """Handler for binary elementwise operations."""
    
    supported_ops = [
        "add", "sub", "mul", "div", "pow",
        "add_", "sub_", "mul_", "div_",
        "__add__", "__sub__", "__mul__", "__truediv__",
        "__radd__", "__rsub__", "__rmul__", "__rtruediv__",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for binary elementwise operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Get second input shape
        input_1_shape = shapes.get("Input_1") or self._get_weight_shape(shapes)
        
        # Normalize op name (remove underscores and dunder)
        op_type = op_name.lower().rstrip("_")
        if op_type.startswith("__"):
            op_type = op_type[2:]
        if op_type.startswith("r"):
            op_type = op_type[1:]  # __radd__ -> add
        
        if input_1_shape is not None:
            return self._generate_binary_elementwise_einsum(
                input_shape, input_1_shape, op_type
            )
        
        # Fallback to unary (scalar broadcast case)
        return self._generate_unary_elementwise_einsum(input_shape, op_type)
    
    def _generate_binary_elementwise_einsum(
        self,
        input_shape: TensorShape,
        input_1_shape: TensorShape,
        op_type: str = "add"
    ) -> EinsumOp:
        """Generate einsum for binary elementwise operations.
        
        Args:
            input_shape: Shape of first input tensor.
            input_1_shape: Shape of second input tensor.
            op_type: Type of binary operation.
            
        Returns:
            EinsumOp for the binary elementwise operation.
        """
        if not input_shape or not input_1_shape:
            raise ValueError(f"Input shapes cannot be empty for {op_type}")
        
        # Handle broadcasting: compute output shape
        max_dims = max(len(input_shape), len(input_1_shape))
        
        # Pad shorter shape with 1s at the front (broadcasting)
        padded_input = [1] * (max_dims - len(input_shape)) + list(input_shape)
        padded_input_1 = [1] * (max_dims - len(input_1_shape)) + list(input_1_shape)
        
        # Compute broadcast output shape
        output_shape = []
        for d1, d2 in zip(padded_input, padded_input_1):
            if d1 == d2 or d1 == 1 or d2 == 1:
                output_shape.append(max(d1, d2))
            else:
                raise ValueError(
                    f"Incompatible shapes for broadcasting: {input_shape} and {input_1_shape}"
                )
        
        # Generate dimension labels
        labels = string.ascii_uppercase[:max_dims]
        
        # Build dimension lists
        input_dims = list(labels)
        input_1_dims = list(labels)
        output_dims = list(labels)
        
        equation = f"{''.join(input_dims)},{''.join(input_1_dims)}->{''.join(output_dims)}"
        
        operands = [
            EinsumOperand("Input", input_dims, is_output=False),
            EinsumOperand("Input_1", input_1_dims, is_output=False),
            EinsumOperand("Output", output_dims, is_output=True),
        ]
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name=op_type,
            is_real_einsum=False,
            elementwise_op=op_type,
            reduction_op="none",
        )
    
    def _generate_unary_elementwise_einsum(
        self,
        shape: TensorShape,
        op_type: str
    ) -> EinsumOp:
        """Generate einsum for scalar broadcast case."""
        dims = len(shape)
        labels = string.ascii_uppercase[:dims]
        
        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Output", list(labels), is_output=True),
        ]
        
        equation = f"{labels}->{labels}"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name=op_type,
            is_real_einsum=False,
            elementwise_op=op_type,
            reduction_op="none",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(UnaryElementwiseHandler)
_registry.register_handler(BinaryElementwiseHandler)


__all__ = ["UnaryElementwiseHandler", "BinaryElementwiseHandler"]

