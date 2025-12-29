"""Handlers for matrix multiplication operations.

This module provides einsum handlers for:
- matmul (matrix multiplication)
- linear (fully connected layer)
- bmm (batch matrix multiplication)
- mm (2D matrix multiplication)
"""

from typing import Any, List

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class MatmulHandler(EinsumOpHandler):
    """Handler for matmul operations."""
    
    supported_ops = ["matmul", "mm"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for matrix multiplication."""
        input_shape = self._get_input_shape(shapes)
        weight_shape = self._get_weight_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # matmul can work without explicit weight shape - infer from input
        if weight_shape is None:
            weight_shape = [input_shape[-1], input_shape[-1]]
        
        return self._generate_matmul_einsum(input_shape, weight_shape)
    
    def _generate_matmul_einsum(
        self,
        input_shape: TensorShape,
        other_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for matrix multiplication.
        
        Args:
            input_shape: Shape of first input tensor.
            other_shape: Shape of second input tensor.
            
        Returns:
            EinsumOp for the matmul operation.
        """
        # Handle different matmul cases
        if len(input_shape) == 2 and len(other_shape) == 2:
            # Standard 2D matmul
            operands = [
                EinsumOperand("Input", ["M", "K"], is_output=False),
                EinsumOperand("Weight", ["K", "N"], is_output=False),
                EinsumOperand("Output", ["M", "N"], is_output=True),
            ]
            equation = "MK,KN->MN"
        else:
            # Batched matmul
            batch_dims = max(len(input_shape), len(other_shape)) - 2
            batch_letters = [f"B{i}" for i in range(batch_dims)]
            dims_a = batch_letters + ["M", "K"]
            dims_b = batch_letters + ["K", "N"]
            dims_out = batch_letters + ["M", "N"]
            
            operands = [
                EinsumOperand("Input", dims_a, is_output=False),
                EinsumOperand("Weight", dims_b, is_output=False),
                EinsumOperand("Output", dims_out, is_output=True),
            ]
            equation = f"{''.join(dims_a)},{''.join(dims_b)}->{''.join(dims_out)}"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="matmul",
            elementwise_op="mul",
            reduction_op="add",
        )


class LinearHandler(EinsumOpHandler):
    """Handler for linear (fully connected) layers."""
    
    supported_ops = ["linear"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for linear layer."""
        input_shape = self._get_input_shape(shapes)
        weight_shape = self._get_weight_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Linear can work without weight shape - use a default based on input
        if weight_shape is None:
            weight_shape = [input_shape[-1], input_shape[-1]]
        
        return self._generate_linear_einsum(input_shape, weight_shape)
    
    def _generate_linear_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for linear layer.
        
        Args:
            input_shape: Shape of input tensor.
            weight_shape: Shape of weight tensor.
            
        Returns:
            EinsumOp for the linear operation.
        """
        # Linear: input @ weight.T
        batch_dims = len(input_shape) - 1
        batch_letters = [f"B{i}" for i in range(batch_dims)]
        
        input_dims = batch_letters + ["K"]
        weight_dims = ["N", "K"]  # Weight is [out_features, in_features]
        output_dims = batch_letters + ["N"]
        
        operands = [
            EinsumOperand("Input", input_dims, is_output=False),
            EinsumOperand("Weight", weight_dims, is_output=False),
            EinsumOperand("Output", output_dims, is_output=True),
        ]
        
        input_str = ''.join(input_dims)
        weight_str = ''.join(weight_dims)
        output_str = ''.join(output_dims)
        equation = f"{input_str},{weight_str}->{output_str}"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="linear",
            elementwise_op="mul",
            reduction_op="add",
        )


class BmmHandler(EinsumOpHandler):
    """Handler for batch matrix multiplication."""
    
    supported_ops = ["bmm"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for batch matrix multiplication."""
        input_shape = self._get_input_shape(shapes)
        weight_shape = self._get_weight_shape(shapes)
        
        if input_shape is None or weight_shape is None:
            raise ValueError(f"Missing Input/Weight shapes for {op_name}")
        
        return self._generate_bmm_einsum(input_shape, weight_shape)
    
    def _generate_bmm_einsum(
        self,
        input_shape: TensorShape,
        other_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for batch matrix multiplication.
        
        bmm: [B, M, K] x [B, K, N] -> [B, M, N]
        """
        operands = [
            EinsumOperand("Input", ["B", "M", "K"], is_output=False),
            EinsumOperand("Weight", ["B", "K", "N"], is_output=False),
            EinsumOperand("Output", ["B", "M", "N"], is_output=True),
        ]
        
        equation = "BMK,BKN->BMN"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name="bmm",
            elementwise_op="mul",
            reduction_op="add",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(MatmulHandler)
_registry.register_handler(LinearHandler)
_registry.register_handler(BmmHandler)


__all__ = ["MatmulHandler", "LinearHandler", "BmmHandler"]

