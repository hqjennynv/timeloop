"""Handlers for convolution operations.

This module provides einsum handlers for:
- conv1d, conv2d, conv3d
- convtranspose1d, convtranspose2d, convtranspose3d
"""

from typing import Any, List, Tuple

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class Conv1dHandler(EinsumOpHandler):
    """Handler for 1D convolution."""
    
    supported_ops = ["conv1d"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 1D convolution."""
        input_shape = self._get_input_shape(shapes)
        weight_shape = self._get_weight_shape(shapes)
        
        if input_shape is None or weight_shape is None:
            raise ValueError(f"Missing Input/Weight shapes for {op_name}")
        
        stride = tuple(kwargs.get("stride", (1,)))
        padding = tuple(kwargs.get("padding", (0,)))
        dilation = tuple(kwargs.get("dilation", (1,)))
        
        return self._generate_conv1d_einsum(
            input_shape, weight_shape, stride, padding, dilation
        )
    
    def _generate_conv1d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int] = (1,),
        padding: Tuple[int] = (0,),
        dilation: Tuple[int] = (1,)
    ) -> EinsumOp:
        """Generate einsum for 1D convolution."""
        B, C, L = input_shape
        O, _, KL = weight_shape
        
        L_out = (L + 2 * padding[0] - dilation[0] * (KL - 1) - 1) // stride[0] + 1
        
        operands = [
            EinsumOperand("Input", ["B", "C", "L"], is_output=False),
            EinsumOperand("Weight", ["O", "C", "K"], is_output=False),
            EinsumOperand("Output", ["B", "O", "LO"], is_output=True),
        ]
        
        equation = "BCL,OCK->BOL"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="conv1d",
            elementwise_op="mul",
            reduction_op="add",
        )


class Conv2dHandler(EinsumOpHandler):
    """Handler for 2D convolution."""
    
    supported_ops = ["conv2d"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 2D convolution."""
        input_shape = self._get_input_shape(shapes)
        weight_shape = self._get_weight_shape(shapes)
        
        if input_shape is None or weight_shape is None:
            raise ValueError(f"Missing Input/Weight shapes for {op_name}")
        
        stride = tuple(kwargs.get("stride", (1, 1)))
        padding = tuple(kwargs.get("padding", (0, 0)))
        dilation = tuple(kwargs.get("dilation", (1, 1)))
        
        return self._generate_conv2d_einsum(
            input_shape, weight_shape, stride, padding, dilation
        )
    
    def _generate_conv2d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1)
    ) -> EinsumOp:
        """Generate einsum for 2D convolution."""
        B, C, H, W = input_shape
        O, _, KH, KW = weight_shape
        
        H_out = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
        
        operands = [
            EinsumOperand("Input", ["B", "C", "H", "W"], is_output=False),
            EinsumOperand("Weight", ["O", "C", "KH", "KW"], is_output=False),
            EinsumOperand("Output", ["B", "O", "HO", "WO"], is_output=True),
        ]
        
        equation = "BCHW,OCKK->BOHW"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="conv2d",
            elementwise_op="mul",
            reduction_op="add",
        )


class Conv3dHandler(EinsumOpHandler):
    """Handler for 3D convolution."""
    
    supported_ops = ["conv3d"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 3D convolution."""
        input_shape = self._get_input_shape(shapes)
        weight_shape = self._get_weight_shape(shapes)
        
        if input_shape is None or weight_shape is None:
            raise ValueError(f"Missing Input/Weight shapes for {op_name}")
        
        stride = tuple(kwargs.get("stride", (1, 1, 1)))
        padding = tuple(kwargs.get("padding", (0, 0, 0)))
        dilation = tuple(kwargs.get("dilation", (1, 1, 1)))
        
        return self._generate_conv3d_einsum(
            input_shape, weight_shape, stride, padding, dilation
        )
    
    def _generate_conv3d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        dilation: Tuple[int, int, int] = (1, 1, 1)
    ) -> EinsumOp:
        """Generate einsum for 3D convolution."""
        B, C, D, H, W = input_shape
        O, _, KD, KH, KW = weight_shape
        
        D_out = (D + 2 * padding[0] - dilation[0] * (KD - 1) - 1) // stride[0] + 1
        H_out = (H + 2 * padding[1] - dilation[1] * (KH - 1) - 1) // stride[1] + 1
        W_out = (W + 2 * padding[2] - dilation[2] * (KW - 1) - 1) // stride[2] + 1
        
        operands = [
            EinsumOperand("Input", ["B", "C", "D", "H", "W"], is_output=False),
            EinsumOperand("Weight", ["O", "C", "KD", "KH", "KW"], is_output=False),
            EinsumOperand("Output", ["B", "O", "DO", "HO", "WO"], is_output=True),
        ]
        
        equation = "BCDHW,OCDKK->BODHW"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="conv3d",
            elementwise_op="mul",
            reduction_op="add",
        )


class ConvTranspose1dHandler(EinsumOpHandler):
    """Handler for 1D transposed convolution."""
    
    supported_ops = ["convtranspose1d", "conv_transpose1d"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 1D transposed convolution."""
        input_shape = self._get_input_shape(shapes)
        weight_shape = self._get_weight_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Generate placeholder weight if missing
        if weight_shape is None:
            c_in = input_shape[1] if len(input_shape) >= 2 else 64
            weight_shape = [c_in, c_in, 3]
        
        return self._generate_convtranspose1d_einsum(input_shape, weight_shape)
    
    def _generate_convtranspose1d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for 1D transposed convolution."""
        operands = [
            EinsumOperand("Input", ["B", "C", "L"], is_output=False),
            EinsumOperand("Weight", ["C", "K", "R"], is_output=False),
            EinsumOperand("Output", ["B", "K", "P"], is_output=True),
        ]
        
        equation = "BCL,CKR->BKP"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="convtranspose1d",
            elementwise_op="mul",
            reduction_op="add",
        )


class ConvTranspose2dHandler(EinsumOpHandler):
    """Handler for 2D transposed convolution."""
    
    supported_ops = ["convtranspose2d", "conv_transpose2d"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 2D transposed convolution."""
        input_shape = self._get_input_shape(shapes)
        weight_shape = self._get_weight_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Generate placeholder weight if missing
        if weight_shape is None:
            c_in = input_shape[1] if len(input_shape) >= 2 else 64
            weight_shape = [c_in, c_in, 3, 3]
        
        return self._generate_convtranspose2d_einsum(input_shape, weight_shape)
    
    def _generate_convtranspose2d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for 2D transposed convolution."""
        operands = [
            EinsumOperand("Input", ["B", "C", "H", "W"], is_output=False),
            EinsumOperand("Weight", ["C", "K", "R", "S"], is_output=False),
            EinsumOperand("Output", ["B", "K", "P", "Q"], is_output=True),
        ]
        
        equation = "BCHW,CKRS->BKPQ"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="convtranspose2d",
            elementwise_op="mul",
            reduction_op="add",
        )


class ConvTranspose3dHandler(EinsumOpHandler):
    """Handler for 3D transposed convolution."""
    
    supported_ops = ["convtranspose3d", "conv_transpose3d"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for 3D transposed convolution."""
        input_shape = self._get_input_shape(shapes)
        weight_shape = self._get_weight_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        # Generate placeholder weight if missing
        if weight_shape is None:
            c_in = input_shape[1] if len(input_shape) >= 2 else 64
            weight_shape = [c_in, c_in, 3, 3, 3]
        
        return self._generate_convtranspose3d_einsum(input_shape, weight_shape)
    
    def _generate_convtranspose3d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for 3D transposed convolution."""
        operands = [
            EinsumOperand("Input", ["B", "C", "D", "H", "W"], is_output=False),
            EinsumOperand("Weight", ["C", "K", "T", "R", "S"], is_output=False),
            EinsumOperand("Output", ["B", "K", "P", "Q", "U"], is_output=True),
        ]
        
        equation = "BCDHW,CKTRS->BKPQU"
        
        return EinsumOp(
            operands=operands, 
            equation=equation, 
            name="convtranspose3d",
            elementwise_op="mul",
            reduction_op="add",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(Conv1dHandler)
_registry.register_handler(Conv2dHandler)
_registry.register_handler(Conv3dHandler)
_registry.register_handler(ConvTranspose1dHandler)
_registry.register_handler(ConvTranspose2dHandler)
_registry.register_handler(ConvTranspose3dHandler)


__all__ = [
    "Conv1dHandler",
    "Conv2dHandler",
    "Conv3dHandler",
    "ConvTranspose1dHandler",
    "ConvTranspose2dHandler",
    "ConvTranspose3dHandler",
]

