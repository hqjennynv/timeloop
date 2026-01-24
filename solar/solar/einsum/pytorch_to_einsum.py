"""Convert PyTorch computation graphs to einsum representation.

This module implements the first stage of the Solar pipeline:

    pytorch_graph.yaml -> einsum_graph.yaml -> einsum_graph_renamed.yaml

The output follows the einsum graph schema:

    layers:
      <layer_id>:
        type: <operation_type>
        einsum_equation: <equation_string>
        elementwise_op: <op>
        reduction_op: <op>
        is_real_einsum: <bool>
        is_einsum_supportable: <bool>
        shapes: {<operand>: <shape>, ...}
        connections: {inputs: [...], outputs: [...]}

Example:
    >>> from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
    >>> converter = PyTorchToEinsum()
    >>> result = converter.convert("input/pytorch_graph.yaml", "output/")
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import yaml
from collections import deque

from solar.common.utils import ensure_directory, NoAliasDumper
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.einsum_rank_renamer import EinsumRankRenamer
from solar.einsum.ops.base import EinsumOp, EinsumOperand, FFNOp, FFNOperand
from solar.einsum.ops.registry import get_global_registry


PathLike = Union[str, Path]

# Operation categories for einsum supportability classification
_REAL_EINSUM_OPS = frozenset({
    "matmul", "mm", "bmm", "linear",
    "conv1d", "conv2d", "conv3d",
    "convtranspose1d", "convtranspose2d", "convtranspose3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "scaled_dot_product_attention", "attention", "sdpa",
    "einsum",
})

_BINARY_ELEMENTWISE_OPS = frozenset({
    "add", "sub", "mul", "div", "pow",
    "add_", "sub_", "mul_", "div_",
    "__add__", "__sub__", "__mul__", "__truediv__",
    "__radd__", "__rsub__", "__rmul__", "__rtruediv__",
})

_UNARY_ELEMENTWISE_OPS = frozenset({
    "relu", "sigmoid", "tanh", "gelu", "selu", "elu", "mish",
    "softmax", "log_softmax", "softplus", "hardswish", "hardsigmoid",
    "abs", "neg", "exp", "log", "sqrt", "rsqrt", "sin", "cos",
    "clamp", "clamp_", "relu_",
    "dropout", "dropout_",
})

_REDUCTION_OPS = frozenset({
    "sum", "mean", "prod", "max", "min", "amax", "amin",
    "argmax", "argmin", "logsumexp", "norm",
})

_NORM_OPS = frozenset({
    "batch_norm", "batchnorm", "batchnorm1d", "batchnorm2d", "batchnorm3d",
    "layer_norm", "layernorm", "group_norm", "groupnorm",
    "instance_norm", "instancenorm", "normalize",
})

_POOLING_OPS = frozenset({
    "max_pool1d", "max_pool2d", "max_pool3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
    "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
})

_SHAPE_OPS = frozenset({
    "view", "reshape", "flatten", "unflatten",
    "squeeze", "unsqueeze", "expand", "repeat",
    "transpose", "permute", "t", "contiguous",
    "cat", "concat", "stack", "split", "chunk",
    "__getitem__", "getitem", "select", "index_select",
})

_MATRIX_OPS = frozenset({"diag", "diagonal", "tril", "triu"})

_EMBEDDING_OPS = frozenset({"embedding"})

_RNN_OPS = frozenset({"gru", "lstm", "rnn"})

_TRIVIAL_OPS = frozenset({
    "clone", "detach", "copy_", "to", "type", "float", "half",
    "hidden-tensor", "output-tensor", "auxiliary-tensor",
    "roll", "pad", "unfold", "fold",
})

_ATTENTION_OPS = frozenset({
    "multi_head_attention_forward", "multihead_attention",
    "flex_attention",
})

_ALL_SUPPORTABLE_OPS = (
    _REAL_EINSUM_OPS | _BINARY_ELEMENTWISE_OPS | _UNARY_ELEMENTWISE_OPS |
    _REDUCTION_OPS | _NORM_OPS | _POOLING_OPS | _SHAPE_OPS |
    _MATRIX_OPS | _EMBEDDING_OPS | _RNN_OPS | _TRIVIAL_OPS | _ATTENTION_OPS
)

_UNSUPPORTABLE_OPS = frozenset({
    "if", "while", "for", "return", "raise",
    "print", "assert", "pass",
})


# FFN yaml dumping with flow style
class FlowDict(dict): pass
class FlowList(list): pass
class LocalDumper(NoAliasDumper): pass
LocalDumper.add_representer(FlowDict, lambda d, x: d.represent_mapping("tag:yaml.org,2002:map", x, flow_style=True))
LocalDumper.add_representer(FlowList, lambda d, x: d.represent_sequence("tag:yaml.org,2002:seq", x, flow_style=True))
def flowify(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "projection":
                out[k] = FlowDict(v) if isinstance(v, dict) else FlowList(v) if isinstance(v, list) else v
            elif k == "tensor_accesses" and isinstance(v, list):
                out[k] = [FlowDict(flowify(t)) if isinstance(t, dict) else flowify(t) for t in v]
            else:
                out[k] = flowify(v)
        return out
    if isinstance(x, list):
        return [flowify(v) for v in x]
    return x


def _product(shape: List[int]) -> int:
    """Compute product of dimensions in a shape.
    
    Args:
        shape: List of dimension sizes.
        
    Returns:
        Product of all dimensions (1 for empty shape).
    """
    result = 1
    for dim in shape:
        result *= int(dim)
    return int(result)


class PyTorchToEinsum:
    """Convert PyTorch computation graphs to einsum representation.
    
    This converter transforms pytorch_graph.yaml files into einsum_graph.yaml
    files, translating PyTorch operations into einsum notation where possible.
    
    Attributes:
        debug: Whether to print debug information.
        enable_agent: Whether to use LLM agent for unknown operations.
        api_key: API key for LLM agent.
        cache_dir: Directory for caching generated handlers.
    """

    def __init__(
        self,
        debug: bool = False,
        enable_agent: bool = False,
        api_key: Optional[str] = None,
        cache_dir: str = "./solar_handlers_cache",
    ) -> None:
        """Initialize the converter.
        
        Args:
            debug: Enable debug output.
            enable_agent: Enable LLM agent for unknown node types.
            api_key: OpenAI API key for LLM agent.
            cache_dir: Directory for caching generated handlers.
        """
        self._debug = debug
        self._enable_agent = enable_agent
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._einsum_analyzer = EinsumAnalyzer(debug=debug)

    @property
    def debug(self) -> bool:
        """Whether debug output is enabled."""
        return self._debug

    @property
    def einsum_analyzer(self) -> EinsumAnalyzer:
        """The einsum analyzer instance."""
        return self._einsum_analyzer

    def convert(
        self,
        pytorch_graph_path: PathLike,
        output_dir: PathLike,
        *,
        copy_graph: bool = True,
        expand_complex_ops: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Convert a PyTorch graph to einsum representation.
        
        This method:
        1. Loads the PyTorch graph
        2. Builds an operation-only graph (collapsing tensor nodes)
        3. Converts operations to einsum notation
        4. Writes einsum_graph.yaml
        5. Renames ranks using BFS and writes einsum_graph_renamed.yaml

        Args:
            pytorch_graph_path: Path to pytorch_graph.yaml (or legacy JSON).
            output_dir: Directory to write output files.
            copy_graph: If True, copy input graph to output directory.
            expand_complex_ops: If True, attempt to expand complex operations.

        Returns:
            The einsum graph dictionary, or None on failure.
        """
        src = Path(pytorch_graph_path)
        out_dir = ensure_directory(output_dir)

        if not src.exists():
            if self._debug:
                print(f"Debug: PyTorch graph not found: {src}")
            return None

        pytorch_graph = self._load_pytorch_graph(src)
        if not pytorch_graph:
            return None

        if copy_graph:
            self._copy_input_graph(src, out_dir, pytorch_graph)

        # Build operation-only graph (collapse tensor nodes)
        op_graph, start_nodes_info = self._build_op_graph(pytorch_graph)

        # Optional complex operation expansion
        if expand_complex_ops:
            op_graph = self._expand_complex_ops(op_graph)

        # Build einsum graph dictionary
        einsum_graph = self._build_einsum_graph(
            pytorch_graph, op_graph, start_nodes_info
        )

        # Write einsum_graph.yaml
        out_path = out_dir / "einsum_graph.yaml"
        with open(out_path, "w") as f:
            yaml.dump(
                einsum_graph, f,
                Dumper=NoAliasDumper,
                sort_keys=False,
                default_flow_style=False
            )

        # Build FFN graph dictionary
        ffn_graph = self._build_ffn_graph(einsum_graph)

        # Write ffn_einsum_graph.yaml
        out_path = out_dir / "ffn_einsum_graph.yaml"
        with open(out_path, "w") as f:
            yaml.dump(
                flowify(ffn_graph), f,
                Dumper=LocalDumper,
                sort_keys=False,
                default_flow_style=False
            )

        if self._debug:
            print(f"✅ Wrote einsum graph: {out_path}")

        # Rename ranks using BFS traversal
        renamer = EinsumRankRenamer(debug=self._debug)
        renamed_path = out_dir / "einsum_graph_renamed.yaml"
        renamer.rename(einsum_graph, renamed_path)

        if self._debug:
            print(f"✅ Wrote renamed einsum graph: {renamed_path}")

        return einsum_graph

    # Backward compatibility alias
    convert_graph = convert

    def _copy_input_graph(
        self,
        src: Path,
        out_dir: Path,
        pytorch_graph: Dict[str, Any],
    ) -> None:
        """Copy input graph to output directory."""
        try:
            dst = out_dir / "pytorch_graph.yaml"
            if src.suffix.lower() in {".yaml", ".yml"}:
                if src.resolve() != dst.resolve():
                    dst.write_text(src.read_text())
            elif not dst.exists():
                with open(dst, "w") as f:
                    yaml.dump(
                        pytorch_graph, f,
                        Dumper=NoAliasDumper,
                        sort_keys=False,
                        default_flow_style=False
                    )
        except Exception:
            if self._debug:
                print("Debug: Failed to copy/write canonical pytorch_graph.yaml")

    def _load_pytorch_graph(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load PyTorch graph from YAML or JSON file.
        
        Args:
            path: Path to the graph file.
            
        Returns:
            The graph dictionary, or None on failure.
        """
        try:
            suffix = path.suffix.lower()
            
            if suffix in {".yaml", ".yml"}:
                with open(path) as f:
                    data = yaml.safe_load(f)
            elif suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
            else:
                if self._debug:
                    print(f"Debug: Unsupported file extension: {path.suffix}")
                return None

            if isinstance(data, dict) and "layers" in data:
                return data
            if isinstance(data, list):
                return self._convert_node_list(data, model_name=path.stem)
                
            if self._debug:
                print(f"Debug: Unexpected structure in {path}")
            return None
            
        except Exception as exc:
            if self._debug:
                print(f"Debug: Failed to load PyTorch graph: {exc}")
            return None

    def _convert_node_list(
        self,
        nodes: List[Dict[str, Any]],
        *,
        model_name: str,
    ) -> Dict[str, Any]:
        """Convert legacy node list format to structured graph dictionary."""
        layers: Dict[str, Any] = {}
        for node in nodes:
            node_id = node.get("node_id") or node.get("name") or "unknown"
            layers[node_id] = {
                "type": node.get("node_type", node.get("type", "unknown")),
                "node_class": node.get("node_class", "UnknownNode"),
                "input_shapes": node.get("input_shapes", []) or [],
                "output_shapes": node.get("output_shapes", []) or [],
                "weight_nodes": node.get("weight_nodes", []) or [],
                "weight_shapes": node.get("weight_shapes", []) or [],
                "module_args": node.get("module_args", {}) or {},
                "connections": {
                    "inputs": node.get("input_nodes", []) or [],
                    "outputs": node.get("output_nodes", []) or [],
                },
            }
        return {"model_name": model_name, "layers": layers}

    def _build_op_graph(
        self,
        pytorch_graph: Dict[str, Any],
    ) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
        """Build operation-only graph by collapsing tensor nodes.
        
        The input PyTorch graph is typically bipartite (TensorNodes and
        Function/Module nodes). This method collapses tensors and connects
        producer operations to consumer operations.
        
        Args:
            pytorch_graph: The PyTorch graph dictionary.
            
        Returns:
            Tuple of (operation graph, start node information).
        """
        layers = pytorch_graph.get("layers") or {}
        tensor_ids, op_ids, auxiliary_ids = self._partition_nodes(layers)

        graph = nx.DiGraph()
        for op_id in op_ids:
            graph.add_node(op_id, **(layers.get(op_id) or {}))

        # Collect auxiliary tensor info for start nodes
        start_nodes_info = self._collect_start_node_info(
            layers, auxiliary_ids, op_ids
        )

        # Connect operations via tensor producers/consumers
        for tensor_id in tensor_ids:
            tensor_data = layers.get(tensor_id) or {}
            conns = tensor_data.get("connections") or {}
            producers = list(conns.get("inputs") or [])
            consumers = list(conns.get("outputs") or [])
            
            for producer in producers:
                for consumer in consumers:
                    if producer in op_ids and consumer in op_ids:
                        if producer != consumer:
                            graph.add_edge(producer, consumer)

        # Fallback: use direct connections if no tensor nodes
        if not tensor_ids:
            for op_id in op_ids:
                conns = (layers.get(op_id) or {}).get("connections") or {}
                outputs = list(conns.get("outputs") or [])
                for out_id in outputs:
                    if out_id in op_ids and out_id != op_id:
                        graph.add_edge(op_id, out_id)

        return graph, start_nodes_info

    def _partition_nodes(
        self,
        layers: Dict[str, Any],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Partition nodes into tensor, operation, and auxiliary categories.
        
        Args:
            layers: The layers dictionary from the PyTorch graph.
            
        Returns:
            Tuple of (tensor_ids, op_ids, auxiliary_tensor_ids).
        """
        tensor_ids: List[str] = []
        op_ids: List[str] = []
        auxiliary_ids: List[str] = []
        
        for node_id, data in (layers or {}).items():
            node_class = (data.get("node_class") or "").lower()
            node_type = (data.get("type") or "").lower()
            
            if node_type == "auxiliary-tensor":
                auxiliary_ids.append(node_id)
            elif "tensornode" in node_class or "tensor" in node_type:
                tensor_ids.append(node_id)
            else:
                op_ids.append(node_id)
                
        return tensor_ids, op_ids, auxiliary_ids

    def _collect_start_node_info(
        self,
        layers: Dict[str, Any],
        auxiliary_ids: List[str],
        op_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Collect information about auxiliary tensors to create start nodes."""
        start_nodes_info: List[Dict[str, Any]] = []
        
        for idx, aux_id in enumerate(sorted(auxiliary_ids)):
            aux_data = layers.get(aux_id) or {}
            conns = aux_data.get("connections") or {}
            output_shapes = aux_data.get("output_shapes") or []
            consumers = list(conns.get("outputs") or [])
            
            # Filter to only include operation nodes
            valid_consumers = [c for c in consumers if c in op_ids]
            
            start_nodes_info.append({
                "original_id": aux_id,
                "index": idx,
                "output_shapes": output_shapes,
                "consumers": valid_consumers,
            })
            
        return start_nodes_info

    def _expand_complex_ops(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Expand complex operations using GraphExpander (best-effort)."""
        if not graph.nodes:
            return graph

        try:
            from solar.einsum.graph_expander import GraphExpander
            
            expander = GraphExpander(
                debug=self._debug,
                enable_agent=self._enable_agent,
                api_key=self._api_key,
                cache_dir=self._cache_dir,
            )
            return expander.expand(graph)
        except Exception:
            return graph

    def _build_einsum_graph(
        self,
        pytorch_graph: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build einsum graph dictionary from operation graph."""
        result: Dict[str, Any] = {
            "model_name": pytorch_graph.get("model_name", "pytorch_model"),
            "layers": {},
        }

        # Add start nodes from auxiliary tensors
        start_node_id_map = self._add_start_nodes(result, start_nodes_info)

        # Convert each operation to einsum representation
        for node_id in op_graph.nodes():
            node_data = dict(op_graph.nodes[node_id] or {})
            layer_dict = self._convert_operation(
                node_id, node_data, op_graph, start_nodes_info, start_node_id_map
            )
            result["layers"][node_id] = layer_dict

        return result

    def _add_start_nodes(
        self,
        result: Dict[str, Any],
        start_nodes_info: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Add start nodes to the einsum graph."""
        start_node_id_map: Dict[str, str] = {}
        
        for info in start_nodes_info:
            idx = info["index"]
            start_id = "start" if idx == 0 else f"start_{idx}"
            original_id = info["original_id"]
            start_node_id_map[original_id] = start_id

            # Build shapes dictionary
            shapes: Dict[str, List[int]] = {}
            output_shapes = info.get("output_shapes") or []
            if output_shapes:
                shapes["Output"] = list(output_shapes[0])
                for i, shape in enumerate(output_shapes[1:], start=1):
                    shapes[f"Output_{i}"] = list(shape)
            
            # Generate einsum equation
            equation = ""
            operands = ""
            if output_shapes and len(output_shapes[0]) > 0:
                dims = len(output_shapes[0])
                labels = string.ascii_uppercase[:dims]
                equation = f"->{labels}"
                operands = {start_id: list(labels)}
            
            result["layers"][start_id] = {
                "type": "start",
                "einsum_equation": equation,
                "elementwise_op": "copy",
                "reduction_op": "none",
                "is_real_einsum": False,
                "is_einsum_supportable": False,
                "shapes": shapes,
                "operands": operands,
                "connections": {
                    "inputs": [],
                    "outputs": info.get("consumers", []),
                },
            }
            
        return start_node_id_map

    def _convert_operation(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        op_graph: nx.DiGraph,
        start_nodes_info: List[Dict[str, Any]],
        start_node_id_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """Convert a single operation to einsum representation."""
        node_type_raw = node_data.get("type", "unknown")
        node_type = self._einsum_analyzer._get_operation_from_name(str(node_type_raw))

        shapes = self._extract_operand_shapes(node_data)
        
        # Get module_args for operations like transpose/permute
        module_args = node_data.get("module_args", {})
        
        # Try to get einsum representation
        equation = ""
        elementwise_op = "mul"
        reduction_op = "add"
        is_real_einsum = True
        is_einsum_supportable = True
        
        try:
            # Pass module_args to the analyzer for transpose/permute operations
            einsum_op = self._einsum_analyzer.get_einsum_op(
                node_type, shapes, module_args=module_args
            )
            equation = einsum_op.equation
            elementwise_op = einsum_op.elementwise_op
            reduction_op = einsum_op.reduction_op
            is_real_einsum = einsum_op.is_real_einsum
            is_einsum_supportable = einsum_op.is_einsum_supportable
            operands = {operand.name: operand.dims for operand in einsum_op.operands}

        except Exception:
            equation = ""
            operands = []
            assert False, f"Failed to get einsum for node {node_id} of type {node_type}"
            is_einsum_supportable = self._is_operation_supportable(node_type)
            
            # Set default ops based on node type
            if node_type in {"add", "sub", "mul", "div"}:
                elementwise_op = node_type
                reduction_op = "none"
                is_real_einsum = False
            elif node_type in {"sum", "mean"}:
                elementwise_op = "copy"
                reduction_op = "add"
                is_real_einsum = False
            elif node_type == "prod":
                elementwise_op = "copy"
                reduction_op = "mul"
                is_real_einsum = False
            elif node_type in {"max", "min"}:
                elementwise_op = "copy"
                reduction_op = node_type
                is_real_einsum = False

        # Build input connections
        input_connections = sorted(list(op_graph.predecessors(node_id)))
        
        # Add start nodes that feed into this operation
        for info in start_nodes_info:
            if node_id in info.get("consumers", []):
                start_id = start_node_id_map.get(info["original_id"])
                if start_id and start_id not in input_connections:
                    input_connections.append(start_id)
        input_connections = sorted(input_connections)
        
        return {
            "type": node_type,
            "einsum_equation": equation,
            "elementwise_op": elementwise_op,
            "reduction_op": reduction_op,
            "is_real_einsum": is_real_einsum,
            "is_einsum_supportable": is_einsum_supportable,
            "shapes": shapes,
            "operands": operands,
            "connections": {
                "inputs": input_connections,
                "outputs": sorted(list(op_graph.successors(node_id))),
            },
        }

    def _extract_operand_shapes(
        self,
        node_data: Dict[str, Any],
    ) -> Dict[str, List[int]]:
        """Extract operand shapes from node data."""
        shapes: Dict[str, List[int]] = {}

        input_shapes = node_data.get("input_shapes") or []
        output_shapes = node_data.get("output_shapes") or []
        weight_shapes = node_data.get("weight_shapes") or []
        weight_nodes = node_data.get("weight_nodes") or []

        # Add input shapes
        if input_shapes:
            shapes["Input"] = list(input_shapes[0])
            for i, shape in enumerate(input_shapes[1:], start=1):
                shapes[f"Input_{i}"] = list(shape)

        # Add output shapes
        if output_shapes:
            shapes["Output"] = list(output_shapes[0])
            for i, shape in enumerate(output_shapes[1:], start=1):
                shapes[f"Output_{i}"] = list(shape)

        # Add weight shapes
        for idx, w_shape in enumerate(weight_shapes):
            name = weight_nodes[idx] if idx < len(weight_nodes) else f"w{idx}"
            key = self._canonical_weight_key(str(name), idx)
            if key in shapes:
                key = f"{key}_{idx}"
            shapes[key] = list(w_shape)

        # Ensure canonical "Weight" entry exists
        if weight_shapes and "Weight" not in shapes:
            first_key = next(
                (k for k in shapes if k.lower().startswith("weight")),
                None
            )
            if first_key:
                shapes["Weight"] = shapes[first_key]

        # Fallback for binary ops
        if "Weight" not in shapes and "Input_1" in shapes:
            shapes["Weight"] = shapes["Input_1"]

        return shapes

    def _canonical_weight_key(self, name: str, idx: int) -> str:
        """Map parameter/buffer names to stable operand keys."""
        raw = name.strip()
        norm = raw.lower()
        
        if norm == "weight":
            return "Weight"
        if norm == "bias":
            return "Bias"
            
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", raw)
        if safe:
            return f"Weight_{safe}"
        return f"Weight_{idx}"

    def _is_operation_supportable(self, op_type: str) -> bool:
        """Check if an operation can be expressed with extended einsum."""
        op = op_type.lower()
        
        # Check against known supportable operations
        if op in _ALL_SUPPORTABLE_OPS:
            return True
            
        # Check for suffixed matches
        for supported_op in _ALL_SUPPORTABLE_OPS:
            if op.endswith(f".{supported_op}"):
                return True
        
        # Check prefixed patterns
        if any(op.startswith(prefix) for prefix in ["torch.", "nn.", "functional."]):
            stripped = op.split(".")[-1]
            return stripped in _ALL_SUPPORTABLE_OPS
        
        # Default: supportable unless explicitly unsupportable
        return op not in _UNSUPPORTABLE_OPS

    def _build_ffn_graph(
        self,
        einsum_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse with topological order to build Fast Fusion graph."""

        # Data structures for FFN graph
        ffn_shapes = dict()
        ffn_einsums = dict()
        ffn_op_names = list() # ensure ordering
        def _add_op(name: str, tensor_accesses: List[FFNOperand], is_copy_operation: bool = False):
            op = FFNOp(name=name, tensor_accesses=tensor_accesses, is_copy_operation=is_copy_operation)
            ffn_op_names.append(name)
            ffn_einsums[name] = op

        # Topological sorting data structures:
        # indegree map and a queue of zero-indegree nodes
        init_nodes = set()
        indegree = dict()
        queue = deque()
        for node_name, node in einsum_graph["layers"].items():
            if len(node["connections"]["inputs"]) == 0:
                init_nodes.add(node_name)
                queue.append(node_name)
            else:
                indegree[node_name] = len(node["connections"]["inputs"])

        # Topological traversal
        while queue:
            node_name = queue.popleft()
            node = einsum_graph["layers"][node_name]

            # Process current node
            if node_name in init_nodes:
                # Input tensors read from memory
                input_dims = node['operands'][node_name]

                input_operand = FFNOperand(name=node_name + "_in", dims_lowercase=input_dims)
                output_operand = FFNOperand(name=node_name, dims_lowercase=input_dims, is_output=True)

                _add_op(name=node_name, tensor_accesses=[input_operand, output_operand], is_copy_operation=True)

            else:
                if len(node['connections']['inputs']) > 2:
                    raise ValueError(f"FFN graph builder only supports unary and binary ops. Inputs given: {node['connections']['inputs']}")

                operands = []
                # Argument #1 in einsum operation
                input_name = node['connections']['inputs'][0]
                input_dims = ffn_einsums[input_name].tensor_accesses[-1].dims_lowercase
                input_eq = node['operands']['Input']
                operands.append(FFNOperand(name=input_name, dims_lowercase=input_eq, dims_uppercase=input_dims))

                if len(node['connections']['inputs']) == 2:
                    # Optional Argument #2 in einsum operation
                    input_name = node['connections']['inputs'][1]
                    input_dims = ffn_einsums[input_name].tensor_accesses[-1].dims_lowercase
                    input_eq = node['operands']['Weight']
                    operands.append(FFNOperand(name=input_name, dims_lowercase=input_eq, dims_uppercase=input_dims))

                # Output of einsum operation
                output_eq = node['operands']['Output']
                operands.append(FFNOperand(name=node_name, dims_lowercase=output_eq, is_output=True))

                _add_op(name=node_name, tensor_accesses=operands)

            # Decrease indegree of neighbors and add to queue if zero
            for neighbor in node["connections"]["outputs"]:
                if neighbor in indegree:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)

        result = {
            "workload": {
                "version": "0.5",
                "shape": ffn_shapes,
                "einsums": [ffn_einsums[name].to_dict() for name in ffn_op_names],
            }
        }
        return result



# Backward compatibility alias
PyTorchEinsumConverter = PyTorchToEinsum


__all__ = [
    "PyTorchToEinsum",
    "PyTorchEinsumConverter",  # Backward compatibility
]

