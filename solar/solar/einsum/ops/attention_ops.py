"""Handlers for attention operations.

This module provides einsum handlers for:
- scaled_dot_product_attention (SDPA)
- flex_attention
- multi_head_attention_forward
"""

from typing import Any, List, Optional

from solar.einsum.ops.base import (
    EinsumOpHandler,
    EinsumOp,
    EinsumOperand,
)
from solar.einsum.ops.registry import get_global_registry
from solar.common.types import ShapeDict, TensorShape


class ScaledDotProductAttentionHandler(EinsumOpHandler):
    """Handler for scaled dot-product attention."""
    
    supported_ops = [
        "scaled_dot_product_attention", "sdpa", "attention",
    ]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for scaled dot-product attention."""
        query = shapes.get("Query") or shapes.get("Input")
        key = shapes.get("Key") or shapes.get("Input_1")
        value = shapes.get("Value") or shapes.get("Input_2")
        
        if not query or not key or not value:
            raise ValueError(f"Missing Q/K/V shapes for attention. Got: {shapes}")
        
        is_causal = kwargs.get("is_causal", False)
        
        return self._generate_sdpa_einsum(query, key, value, is_causal)
    
    def _generate_sdpa_einsum(
        self,
        query_shape: TensorShape,
        key_shape: TensorShape,
        value_shape: TensorShape,
        is_causal: bool = False
    ) -> EinsumOp:
        """Generate einsum for scaled dot-product attention.
        
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        Args:
            query_shape: [batch, heads, seq_q, d_k]
            key_shape: [batch, heads, seq_k, d_k]
            value_shape: [batch, heads, seq_v, d_v]
            is_causal: Whether causal masking is applied
        """
        operands = [
            EinsumOperand("Query", ["B", "H", "Q", "D"], is_output=False),
            EinsumOperand("Key", ["B", "H", "K", "D"], is_output=False),
            EinsumOperand("Value", ["B", "H", "K", "V"], is_output=False),
            EinsumOperand("Output", ["B", "H", "Q", "V"], is_output=True),
        ]
        
        equation = "BHQD,BHKD,BHKV->BHQV"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name="scaled_dot_product_attention",
            elementwise_op="mul",
            reduction_op="add",
        )


class FlexAttentionHandler(EinsumOpHandler):
    """Handler for flex_attention (similar to SDPA)."""
    
    supported_ops = ["flex_attention"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for flex_attention."""
        query = shapes.get("Query") or shapes.get("Input")
        key = shapes.get("Key") or shapes.get("Input_1")
        value = shapes.get("Value") or shapes.get("Input_2")
        
        if not query or not key or not value:
            raise ValueError(f"Missing Q/K/V shapes for attention. Got: {shapes}")
        
        return self._generate_flex_attention_einsum(query, key, value)
    
    def _generate_flex_attention_einsum(
        self,
        query_shape: TensorShape,
        key_shape: TensorShape,
        value_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for flex_attention."""
        operands = [
            EinsumOperand("Query", ["B", "H", "Q", "D"], is_output=False),
            EinsumOperand("Key", ["B", "H", "K", "D"], is_output=False),
            EinsumOperand("Value", ["B", "H", "K", "V"], is_output=False),
            EinsumOperand("Output", ["B", "H", "Q", "V"], is_output=True),
        ]
        
        equation = "BHQD,BHKD,BHKV->BHQV"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name="flex_attention",
            elementwise_op="mul",
            reduction_op="add",
        )


class MultiHeadAttentionHandler(EinsumOpHandler):
    """Handler for multi-head attention forward."""
    
    supported_ops = ["multi_head_attention_forward", "multihead_attention"]
    
    def generate_einsum(
        self,
        op_name: str,
        shapes: ShapeDict,
        **kwargs: Any
    ) -> EinsumOp:
        """Generate einsum for multi-head attention."""
        input_shape = self._get_input_shape(shapes)
        
        if input_shape is None:
            raise ValueError(f"Missing Input shape for {op_name}")
        
        return self._generate_mha_einsum(input_shape)
    
    def _generate_mha_einsum(self, input_shape: TensorShape) -> EinsumOp:
        """Generate einsum for multi-head attention.
        
        MHA combines Q, K, V projections with attention computation.
        We represent it as a single attention-like operation.
        """
        operands = [
            EinsumOperand("Input", ["B", "S", "D"], is_output=False),
            EinsumOperand("Output", ["B", "S", "D"], is_output=True),
        ]
        
        equation = "BSD->BSD"
        
        return EinsumOp(
            operands=operands,
            equation=equation,
            name="multi_head_attention_forward",
            is_real_einsum=False,  # Composite operation
            elementwise_op="mul",
            reduction_op="add",
        )


# Register handlers with global registry (without loading other handlers)
_registry = get_global_registry(load_handlers=False)
_registry.register_handler(ScaledDotProductAttentionHandler)
_registry.register_handler(FlexAttentionHandler)
_registry.register_handler(MultiHeadAttentionHandler)


__all__ = [
    "ScaledDotProductAttentionHandler",
    "FlexAttentionHandler",
    "MultiHeadAttentionHandler",
]

