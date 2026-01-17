"""Base classes for einsum operation handlers.

This module defines the core data structures and abstract base class
for all einsum operation handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from operator import eq
from typing import Any, Dict, List, Optional, Tuple, Union

from solar.common.types import ShapeDict, TensorShape


@dataclass
class EinsumOperand:
    """Represents an operand in an einsum operation."""
    name: str
    dims: List[str]
    is_output: bool = False
    stride: Optional[Dict[str, int]] = None
    dilation: Optional[Dict[str, int]] = None
    
    def to_timeloop_dataspace(self) -> Dict[str, Any]:
        """Convert to timeloop dataspace format."""
        dataspace = {
            'name': self.name,
            'projection': self.dims,
        }
        if self.is_output:
            dataspace['read_write'] = 'true'
        return dataspace


@dataclass
class EinsumOp:
    """Represents an einsum operation.
    
    The extended einsum representation supports different elementwise and reduction
    operations beyond the standard multiply-add semantics:
    
    - elementwise_op: The operation applied element-wise (default: 'mul')
      Examples: 'mul' for matmul, 'add' for element-wise add, 'max' for max pooling
    - reduction_op: The operation used to reduce/aggregate (default: 'add')
      Examples: 'add' for sum, 'max' for max reduction, 'none' for no reduction
    - is_real_einsum: True if this is a standard tensor contraction (mul+add)
    - is_einsum_supportable: True if the operation can be expressed with extended einsum
    
    Standard einsum (matmul): elementwise_op='mul', reduction_op='add'
    Element-wise add:        elementwise_op='add', reduction_op='none'
    Max pooling:             elementwise_op='copy', reduction_op='max'
    """
    operands: List[EinsumOperand]
    equation: str
    name: str
    is_real_einsum: bool = True
    elementwise_op: str = "mul"  # 'mul', 'add', 'sub', 'div', 'max', 'min', 'copy'
    reduction_op: str = "add"    # 'add', 'max', 'min', 'mul', 'none'
    is_einsum_supportable: bool = True  # Can this op be expressed with extended einsum?
    
    @property
    def input_operands(self) -> List[EinsumOperand]:
        """Get input operands."""
        return [op for op in self.operands if not op.is_output]
    
    @property
    def output_operands(self) -> List[EinsumOperand]:
        """Get output operands."""
        return [op for op in self.operands if op.is_output]
    
    def get_compute_cost(self, shapes: ShapeDict) -> int:
        """Calculate compute cost for the operation."""
        # Special-cases for operations where a direct formula is clearer and more
        # accurate than deriving cost from symbolic einsum dims.
        if self.name in {"conv1d", "conv2d", "conv3d"}:
            weight = shapes.get("Weight")
            output = shapes.get("Output")
            if weight and output:
                # Weight shapes:
                # - conv1d: [O, C, K]
                # - conv2d: [O, C, KH, KW]
                # - conv3d: [O, C, KD, KH, KW]
                out_elems = 1
                for d in output:
                    out_elems *= d
                in_channels = int(weight[1]) if len(weight) >= 2 else 1
                kernel_elems = 1
                for d in weight[2:]:
                    kernel_elems *= d
                return out_elems * in_channels * kernel_elems

        # Generic fallback: multiply unique ranks across *input* operands only.
        # This keeps tests/user code simple (no need to provide Output shapes)
        # while remaining correct for matmul/linear/elementwise/reduction-style ops.
        all_ranks: Dict[str, int] = {}
        for op in self.input_operands:
            if op.name not in shapes:
                raise ValueError(f"Shape not found for operand {op.name}")
            shape = shapes[op.name]
            for idx, dim in enumerate(op.dims):
                if dim not in all_ranks:
                    all_ranks[dim] = int(shape[idx])
        
        total_ops = 1
        for rank in all_ranks.values():
            total_ops *= rank
        return int(total_ops)
    
    def to_torch_einsum(self, tensor_names: Optional[List[str]] = None) -> str:
        """Convert to torch.einsum format."""
        input_operands = self.input_operands
        
        if tensor_names is None:
            tensor_names = [op.name for op in input_operands]
        elif len(tensor_names) != len(input_operands):
            raise ValueError(
                f"Number of tensor names ({len(tensor_names)}) must match "
                f"number of input operands ({len(input_operands)})"
            )
        
        equation_str = f"'{self.equation}'"
        tensor_args = ', '.join(tensor_names)
        return f"torch.einsum({equation_str}, {tensor_args})"


class EinsumOpHandler(ABC):
    """Abstract base class for einsum operation handlers.
    
    Each handler is responsible for converting one or more related operation
    types to einsum notation.
    """
    
    # Class attribute: list of operation names this handler supports
    supported_ops: List[str] = []
    
    def __init__(self, debug: bool = False):
        """Initialize the handler.
        
        Args:
            debug: Enable debug output.
        """
        self.debug = debug
    
    @abstractmethod
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate an einsum operation for the given operation.
        
        Args:
            op_name: Normalized operation name.
            shapes: Dictionary of tensor shapes.
            **kwargs: Additional operation-specific parameters.
            
        Returns:
            EinsumOp representing the operation.
            
        Raises:
            ValueError: If operation cannot be converted.
        """
        pass
    
    def can_handle(self, op_name: str) -> bool:
        """Check if this handler can process the given operation.
        
        Args:
            op_name: Normalized operation name.
            
        Returns:
            True if this handler supports the operation.
        """
        return op_name.lower() in [op.lower() for op in self.supported_ops]
    
    def _get_input_shape(self, shapes: ShapeDict) -> Optional[TensorShape]:
        """Get input shape from shapes dict."""
        return shapes.get("Input") or shapes.get("input")
    
    def _get_weight_shape(self, shapes: ShapeDict) -> Optional[TensorShape]:
        """Get weight shape from shapes dict."""
        return shapes.get("Weight") or shapes.get("weight")
    
    def _get_output_shape(self, shapes: ShapeDict) -> Optional[TensorShape]:
        """Get output shape from shapes dict."""
        return shapes.get("Output") or shapes.get("output")

@dataclass
class FFNOperand:
    """Represents an operand in Fast Fusion (FFN) einsum format."""
    name: str
    dims_lowercase: List[str]
    dims_uppercase: Optional[List[str]] = None
    is_output: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict to finally dump to FFN yaml."""
        operand = dict()
        operand['name'] = self.name
        operand['projection'] = (
            [d.lower() for d in self.dims_lowercase]
            if self.dims_uppercase is None
            else {d.upper(): pr.lower() for d, pr in zip(self.dims_uppercase, self.dims_lowercase)}
        )
        if self.is_output:
            operand['output'] = True
        return operand

@dataclass
class FFNOp:
    """Represents an operation in Fast Fusion (FFN) einsum format."""
    name: str
    tensor_accesses: List[FFNOperand]
    is_copy_operation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict to finally dump to FFN yaml."""
        op_dict = dict()
        op_dict['name'] = self.name
        if self.is_copy_operation:
            op_dict['is_copy_operation'] = True
        op_dict['tensor_accesses'] = [operand.to_dict() for operand in self.tensor_accesses]
        return op_dict


__all__ = ["EinsumOperand", "EinsumOp", "EinsumOpHandler", "FFNOperand", "FFNOp"]

