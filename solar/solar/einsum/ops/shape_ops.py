"""Handlers for shape/view operations.

This module provides einsum handlers for:
- view, reshape, flatten, unflatten
- squeeze, unsqueeze, expand, repeat
- transpose, permute, t, contiguous
- cat, concat, stack, split, chunk
- getitem, select, index_select
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


def generate_dim_labels(num_dims: int, prefix: str = "") -> List[str]:
    """Generate dimension labels that support more than 26 dimensions.
    
    Uses a char+number scheme:
    - First 26 dims: A, B, C, ..., Z (or with prefix: i0, i1, ..., i25)
    - Beyond 26: A0, A1, ..., Z0, Z1, ... (or i26, i27, ...)
    
    Args:
        num_dims: Number of dimension labels to generate.
        prefix: Optional prefix for labels (e.g., "i" for input, "o" for output).
        
    Returns:
        List of dimension label strings.
    """
    if prefix:
        # Use prefix + number scheme: i0, i1, i2, ... or o0, o1, o2, ...
        return [f"{prefix}{i}" for i in range(num_dims)]
    
    # Use letter-based scheme with numbers for overflow
    labels = []
    for i in range(num_dims):
        if i < 26:
            # First 26: A, B, C, ..., Z
            labels.append(string.ascii_uppercase[i])
        else:
            # Beyond 26: A0, A1, ..., Z0, Z1, A2, ...
            letter_idx = (i - 26) % 26
            number = (i - 26) // 26
            labels.append(f"{string.ascii_uppercase[letter_idx]}{number}")
    return labels


class TensorManipulationHandler(EinsumOpHandler):
    """Handler for tensor manipulation operations."""
    
    supported_ops = [
        "view", "reshape", "flatten", "unflatten",
        "squeeze", "unsqueeze", "expand", "repeat",
        "transpose", "permute", "t", "contiguous",
        "cat", "concat", "stack", "split", "chunk",
        "__getitem__", "getitem", "select", "index_select",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for tensor manipulation operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        output_shape = self._get_output_shape(shapes) or input_shape
        
        return self._generate_tensor_manipulation_einsum(
            input_shape, output_shape, op_name
        )
    
    def _generate_tensor_manipulation_einsum(
        self,
        input_shape: TensorShape,
        output_shape: TensorShape,
        operation_name: str
    ) -> EinsumOp:
        """Generate einsum for tensor manipulation.
        
        For view/reshape operations, input and output use different dimension
        variables since they represent different tensor layouts:
        - Input dims: i0, i1, i2, ...
        - Output dims: o0, o1, o2, ...
        
        Example: view [2,32,64] -> [2,32,4,16]
        - Input: i0,i1,i2 (3 dims)
        - Output: o0,o1,o2,o3 (4 dims)
        - Equation: i0i1i2->o0o1o2o3
        """
        in_dims = len(input_shape)
        out_dims = len(output_shape)
        
        # Use prefix-based labels for unlimited dimension support
        # Input uses "i" prefix, output uses "o" prefix
        in_label_list = generate_dim_labels(in_dims, prefix="I")
        out_label_list = generate_dim_labels(out_dims, prefix="O")
        
        # Join labels for equation string
        in_labels = "".join(in_label_list)
        out_labels = "".join(out_label_list)
        
        operands = [
            EinsumOperand("Input", in_label_list, is_output=False),
            EinsumOperand("Output", out_label_list, is_output=True),
        ]
        
        equation = f"{in_labels}->{out_labels}"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name=operation_name,
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op="none",
        )


class MatrixStructureHandler(EinsumOpHandler):
    """Handler for matrix structure operations."""
    
    supported_ops = ["diag", "diagonal", "tril", "triu"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for matrix structure operation."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        return self._generate_matrix_structure_einsum(input_shape, op_name)
    
    def _generate_matrix_structure_einsum(
        self,
        input_shape: TensorShape,
        operation: str
    ) -> EinsumOp:
        """Generate einsum for matrix structure ops."""
        dims = len(input_shape)
        labels = string.ascii_uppercase[:dims]
        
        if operation == "diag":
            if dims == 1:
                # 1D -> 2D diagonal matrix
                output_labels = "AA"
            else:
                # 2D+ -> 1D diagonal extraction
                output_labels = labels[0]
        elif operation == "transpose" and dims >= 2:
            # Swap last two dims
            output_labels = labels[:-2] + labels[-1] + labels[-2]
        else:
            # tril, triu preserve shape
            output_labels = labels
        
        operands = [
            EinsumOperand("Input", list(labels), is_output=False),
            EinsumOperand("Output", list(output_labels), is_output=True),
        ]
        
        equation = f"{labels}->{output_labels}"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name=operation,
            is_real_einsum=False,
            elementwise_op="copy",
            reduction_op="none",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(TensorManipulationHandler)
_registry.register_handler(MatrixStructureHandler)


__all__ = ["TensorManipulationHandler", "MatrixStructureHandler"]

