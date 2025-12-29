"""Core einsum analyzer for converting operations to einsum notation.

This module provides the EinsumAnalyzer class which uses the registered
operation handlers to convert PyTorch operations to einsum notation.
"""

import string
from typing import Any, Dict, List, Optional, Tuple

from solar.common.types import ShapeDict, TensorShape
from solar.einsum.ops.base import EinsumOp, EinsumOperand
from solar.einsum.ops.registry import get_global_registry, EinsumOpRegistry


class EinsumAnalyzer:
    """Analyzes operations and converts them to einsum notation.
    
    This class provides methods to convert various PyTorch operations
    (matmul, conv, attention, etc.) to einsum notation for analysis.
    
    The actual conversion logic is delegated to handlers registered
    in the global EinsumOpRegistry.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the EinsumAnalyzer.
        
        Args:
            debug: Enable debug output.
        """
        self.debug = debug
        self._registry = get_global_registry()
    
    def get_compute_cost(self, op_name: str, shapes: ShapeDict, **kwargs: Any) -> int:
        """Get compute cost for an operation.
        
        Args:
            op_name: Name of the operation.
            shapes: Dictionary of tensor shapes.
            
        Returns:
            Number of operations required.
        """
        shapes_for_cost: ShapeDict = dict(shapes)

        op_norm = self._get_operation_from_name(op_name)
        if op_norm in {"conv1d", "conv2d", "conv3d"}:
            # If output shape is not provided, derive it from input/weight and kwargs.
            if "Output" not in shapes_for_cost and "Input" in shapes_for_cost and "Weight" in shapes_for_cost:
                input_shape = shapes_for_cost["Input"]
                weight_shape = shapes_for_cost["Weight"]
                out_shape = self._infer_conv_output_shape(op_norm, input_shape, weight_shape, **kwargs)
                if out_shape:
                    shapes_for_cost["Output"] = out_shape

        einsum_op = self.get_einsum_op(op_name, shapes_for_cost, **kwargs)
        return einsum_op.get_compute_cost(shapes_for_cost)
    
    def get_memory_cost(self, shapes: ShapeDict) -> Dict[str, int]:
        """Calculate memory cost for tensors.
        
        Args:
            shapes: Dictionary of tensor shapes.
            
        Returns:
            Dictionary mapping tensor names to element counts.
        """
        memory_cost: Dict[str, int] = {}
        for name, shape in shapes.items():
            elements = 1
            for dim in shape:
                elements *= dim
            memory_cost[name] = elements
        memory_cost["total"] = sum(memory_cost.values())
        return memory_cost
    
    def get_einsum_op(self, op_name: str, shapes: ShapeDict, **kwargs: Any) -> EinsumOp:
        """Get an einsum operation for the given operation name.
        
        Args:
            op_name: Name of the operation.
            shapes: Dictionary of tensor shapes.
            
        Returns:
            EinsumOp object.
            
        Raises:
            ValueError: If operation is not supported.
        """
        op_norm = self._get_operation_from_name(op_name)
        
        # Try to get handler from registry
        if self._registry.has_handler(op_norm):
            return self._registry.get_einsum_op(op_norm, shapes, **kwargs)
        
        # Fallback: if we have input_shape, treat as generic elementwise
        input_shape = shapes.get("Input") or shapes.get("input")
        if input_shape is not None:
            dims = len(input_shape)
            labels = string.ascii_uppercase[:dims]
            
            operands = [
                EinsumOperand("Input", list(labels), is_output=False),
                EinsumOperand("Output", list(labels), is_output=True),
            ]
            
            equation = f"{labels}->{labels}"
            
            return EinsumOp(
                operands=operands, 
                equation=equation, 
                name=op_norm,
                is_real_einsum=False,
                elementwise_op="copy",
                reduction_op="none",
            )
        
        raise ValueError(f"Unsupported operation: {op_name}")
    
    def _get_operation_from_name(self, op_name: str) -> str:
        """Normalize an operation name to a canonical operation key."""
        op = op_name.lower()

        # Common namespace prefixes.
        if op.startswith("torch.nn."):
            op = op[len("torch.nn.") :]
        if op.startswith("torch."):
            op = op[len("torch.") :]

        # Transpose Convolutions (check first since they contain conv1d/2d/3d)
        if "convtranspose1d" in op or "conv_transpose1d" in op:
            return "convtranspose1d"
        if "convtranspose2d" in op or "conv_transpose2d" in op:
            return "convtranspose2d"
        if "convtranspose3d" in op or "conv_transpose3d" in op:
            return "convtranspose3d"
        
        # Regular Convolutions
        if "conv1d" in op:
            return "conv1d"
        if "conv2d" in op:
            return "conv2d"
        if "conv3d" in op:
            return "conv3d"

        # Linear/matmul.
        if "linear" in op:
            return "linear"
        if "matmul" in op or op in {"mm", "bmm"}:
            if op == "bmm":
                return "bmm"
            return "matmul"

        # Binary elementwise operations
        if op == "add" or op.endswith(".add") or op.endswith("_add"):
            return "add"
        if op == "sub" or op.endswith(".sub") or op.endswith("_sub"):
            return "sub"
        if op == "mul" or op.endswith(".mul") or op.endswith("_mul"):
            return "mul"
        if op == "div" or op.endswith(".div") or op.endswith("_div"):
            return "div"

        # Unary elementwise activations
        if "relu" in op:
            return "relu"
        if "sigmoid" in op:
            return "sigmoid"
        if "tanh" in op:
            return "tanh"
        if "gelu" in op:
            return "gelu"
        if "softmax" in op:
            return "softmax"
        if op in {"abs", "neg", "exp", "log", "sqrt", "rsqrt", "sin", "cos"}:
            return op

        # Reductions.
        if "sum" in op and "logsumexp" not in op:
            return "sum"
        if "mean" in op:
            return "mean"
        if "prod" in op:
            return "prod"
        if op in {"max", "amax"} or op.endswith(".max"):
            return "max"
        if op in {"min", "amin"} or op.endswith(".min"):
            return "min"

        # Fallback: last path component.
        return op.split(".")[-1]

    def get_reduction_einsum_op(
        self,
        op_name: str,
        shapes: ShapeDict,
        reduce_dims: Optional[List[int]] = None,
        keepdim: bool = False,
    ) -> EinsumOp:
        """Get an einsum op for a reduction (sum/mean/prod)."""
        op_norm = self._get_operation_from_name(op_name)
        return self.get_einsum_op(op_norm, shapes, dims=reduce_dims, keepdim=keepdim)

    def _infer_conv_output_shape(
        self,
        op_norm: str,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        **kwargs: Any,
    ) -> Optional[TensorShape]:
        """Infer output shape for conv ops when not provided."""
        try:
            if op_norm == "conv1d":
                b, _c, l = input_shape
                o, _c2, k = weight_shape
                stride = int((kwargs.get("stride") or (1,))[0])
                padding = int((kwargs.get("padding") or (0,))[0])
                dilation = int((kwargs.get("dilation") or (1,))[0])
                l_out = (l + 2 * padding - dilation * (k - 1) - 1) // stride + 1
                return [b, o, l_out]

            if op_norm == "conv3d":
                b, _c, d, h, w = input_shape
                o, _c2, kd, kh, kw = weight_shape
                stride = tuple(kwargs.get("stride") or (1, 1, 1))
                padding = tuple(kwargs.get("padding") or (0, 0, 0))
                dilation = tuple(kwargs.get("dilation") or (1, 1, 1))
                d_out = (d + 2 * padding[0] - dilation[0] * (kd - 1) - 1) // stride[0] + 1
                h_out = (h + 2 * padding[1] - dilation[1] * (kh - 1) - 1) // stride[1] + 1
                w_out = (w + 2 * padding[2] - dilation[2] * (kw - 1) - 1) // stride[2] + 1
                return [b, o, d_out, h_out, w_out]

            # Default conv2d.
            b, _c, h, w = input_shape
            o, _c2, kh, kw = weight_shape
            stride = tuple(kwargs.get("stride") or (1, 1))
            padding = tuple(kwargs.get("padding") or (0, 0))
            dilation = tuple(kwargs.get("dilation") or (1, 1))
            h_out = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
            w_out = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
            return [b, o, h_out, w_out]
        except Exception:
            return None
    
    def get_torch_einsum_equation(
        self,
        op_name: str,
        shapes: Optional[ShapeDict] = None
    ) -> str:
        """Get torch einsum equation string for an operation.
        
        Args:
            op_name: Name of the operation.
            shapes: Optional dictionary of tensor shapes.
            
        Returns:
            Einsum equation string.
        """
        if not shapes:
            # Return generic equation based on operation type
            op_lower = op_name.lower()
            if 'matmul' in op_lower:
                return "ij,jk->ik"
            elif 'linear' in op_lower:
                return "...k,nk->...n"
            elif 'conv2d' in op_lower:
                return "bchw,ockk->bohw"
            else:
                return ""
        
        einsum_op = self.get_einsum_op(op_name, shapes)
        return einsum_op.equation
    
    # =========================================================================
    # Backward compatibility methods
    # =========================================================================
    
    def generate_matmul_einsum(self, input_shape: TensorShape, other_shape: TensorShape) -> EinsumOp:
        """Generate einsum for matrix multiplication (backward compatibility)."""
        return self.get_einsum_op("matmul", {"Input": input_shape, "Weight": other_shape})
    
    def generate_linear_einsum(self, input_shape: TensorShape, weight_shape: TensorShape) -> EinsumOp:
        """Generate einsum for linear layer (backward compatibility)."""
        return self.get_einsum_op("linear", {"Input": input_shape, "Weight": weight_shape})
    
    def generate_conv2d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1)
    ) -> EinsumOp:
        """Generate einsum for 2D convolution (backward compatibility)."""
        return self.get_einsum_op(
            "conv2d",
            {"Input": input_shape, "Weight": weight_shape},
            stride=stride, padding=padding, dilation=dilation
        )
    
    def generate_conv1d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int] = (1,),
        padding: Tuple[int] = (0,),
        dilation: Tuple[int] = (1,)
    ) -> EinsumOp:
        """Generate einsum for 1D convolution (backward compatibility)."""
        return self.get_einsum_op(
            "conv1d",
            {"Input": input_shape, "Weight": weight_shape},
            stride=stride, padding=padding, dilation=dilation
        )
    
    def generate_conv3d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        dilation: Tuple[int, int, int] = (1, 1, 1)
    ) -> EinsumOp:
        """Generate einsum for 3D convolution (backward compatibility)."""
        return self.get_einsum_op(
            "conv3d",
            {"Input": input_shape, "Weight": weight_shape},
            stride=stride, padding=padding, dilation=dilation
        )
    
    def generate_elementwise_einsum(self, shape: TensorShape, op_type: str = "elementwise") -> EinsumOp:
        """Generate einsum for elementwise operations (backward compatibility)."""
        return self.get_einsum_op(op_type, {"Input": shape})
    
    def generate_binary_elementwise_einsum(
        self,
        input_shape: TensorShape,
        input_1_shape: TensorShape,
        op_type: str = "add"
    ) -> EinsumOp:
        """Generate einsum for binary elementwise operations (backward compatibility)."""
        return self.get_einsum_op(op_type, {"Input": input_shape, "Input_1": input_1_shape})
    
    def generate_reduction_einsum(
        self,
        shape: TensorShape,
        op_type: str = "sum",
        dims: Optional[List[int]] = None,
        keepdim: bool = False
    ) -> EinsumOp:
        """Generate einsum for reduction operations (backward compatibility)."""
        return self.get_einsum_op(op_type, {"Input": shape}, dims=dims, keepdim=keepdim)


__all__ = ["EinsumAnalyzer"]

