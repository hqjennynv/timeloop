"""TorchView graph processor for extracting layer information from PyTorch models.

This module provides functionality to process torchview ComputationGraph objects
and extract detailed information about model layers, following Google's Python
style guide.

The output format matches the original process_torchview_graph.py output:
- node_id: Hierarchical node identifier (e.g., "Model.linear_0")
- node_type: Operation type (e.g., "linear", "conv2d", "matmul")
- node_class: Actual node class (e.g., "FunctionNode", "TensorNode", "ModuleNode")
- input_nodes: List of input node IDs (connections from predecessors)
- output_nodes: List of output node IDs (connections to successors)
- input_shapes: List of input tensor shapes
- output_shapes: List of output tensor shapes
- weight_nodes: List of parameter names (e.g., ["weight", "bias"])
- weight_shapes: List of parameter shapes
- module_args: Dictionary of module configuration arguments
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch import nn

from solar.common.constants import (
    BOOLEAN_ATTRS,
    GEOMETRIC_ATTRS,
    MODULE_ATTR_NAMES,
)
from solar.common.types import GraphInfo, NodeInfo, ShapeDict, TensorShape
from solar.common.utils import ensure_directory


class TorchviewProcessor:
    """Processes torchview computation graphs to extract layer information.
    
    This class provides methods to extract detailed information from torchview
    ComputationGraph objects, including node hierarchies, shapes, connections,
    and PyTorch nn.Module parameters.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the TorchviewProcessor.
        
        Args:
            debug: Enable debug output for troubleshooting.
        """
        self.debug = debug
        self._processed_nodes: set = set()
        self._matched_modules: set = set()
        self._node_counter: Dict[str, int] = {}
        # Mapping from original node object id to clean node_id
        self._original_to_clean_id: Dict[str, str] = {}
    
    def process_graph(self,
                     computation_graph: Any,
                     output_dir: str,
                     kernel_name: str,
                     original_model: Optional[nn.Module] = None) -> List[NodeInfo]:
        """Process a torchview ComputationGraph and save extracted layer nodes.
        
        Args:
            computation_graph: torchview ComputationGraph object.
            output_dir: Directory to save outputs.
            kernel_name: Name of the kernel for file naming.
            original_model: Original PyTorch model for parameter extraction.
            
        Returns:
            List of NodeInfo objects containing extracted layer information.
        """
        if self.debug:
            print(f"Processing torchview graph for {kernel_name}...")
        
        # Reset state for new graph
        self._reset_state()
        
        # Extract layer nodes
        layer_nodes = self._extract_layer_nodes(computation_graph, original_model)
        
        # Save canonical YAML graph (and remove any legacy JSON artifacts).
        output_path = Path(output_dir)
        ensure_directory(output_path)
        yaml_filename = output_path / "pytorch_graph.yaml"

        self._save_pytorch_graph_yaml(layer_nodes, yaml_filename, model_name=kernel_name)
        
        if self.debug:
            self._print_layer_summary(layer_nodes)
        
        return layer_nodes
    
    def _reset_state(self) -> None:
        """Reset internal state for processing a new graph."""
        self._processed_nodes.clear()
        self._matched_modules.clear()
        self._node_counter.clear()
        self._original_to_clean_id.clear()
    
    def _extract_layer_nodes(self,
                           computation_graph: Any,
                           original_model: Optional[nn.Module] = None) -> List[NodeInfo]:
        """Extract layer nodes from the computation graph.
        
        Args:
            computation_graph: torchview ComputationGraph object.
            original_model: Original PyTorch model for parameter extraction.
            
        Returns:
            List of NodeInfo objects.
        """
        # Try different extraction methods in order of preference
        layer_nodes = []
        
        # Method 1: Extract from node hierarchy if available
        if hasattr(computation_graph, 'node_hierarchy') and computation_graph.node_hierarchy:
            if self._is_hierarchy_useful(computation_graph.node_hierarchy):
                if self.debug:
                    print("Extracting from node_hierarchy...")
                return self._extract_from_hierarchy(
                    computation_graph.node_hierarchy, 'Model'
                )
        
        # Method 2: Extract from edge_list (most common path)
        if hasattr(computation_graph, 'edge_list') and computation_graph.edge_list:
            if self.debug:
                print(f"Extracting from edge_list ({len(computation_graph.edge_list)} edges)...")
            layer_nodes = self._extract_from_edge_list(computation_graph, original_model)
            return layer_nodes
        
        # Method 3: Parse visual graph as fallback
        if hasattr(computation_graph, 'visual_graph'):
            if self.debug:
                print("Parsing visual graph...")
            return self._extract_from_visual_graph(computation_graph.visual_graph)
        
        return layer_nodes
    
    def _is_hierarchy_useful(self, node_hierarchy: Dict[str, Any]) -> bool:
        """Check if node hierarchy contains useful computation nodes.
        
        Args:
            node_hierarchy: Node hierarchy dictionary.
            
        Returns:
            True if hierarchy contains useful nodes, False otherwise.
        """
        for key, node in node_hierarchy.items():
            node_class = type(node).__name__
            if node_class in ['TensorNode', 'ModuleNode', 'FunctionNode']:
                return True
        return False
    
    def _extract_from_hierarchy(self,
                              node_hierarchy: Dict[str, Any],
                              parent_name: str = '') -> List[NodeInfo]:
        """Recursively extract nodes from hierarchy.
        
        Args:
            node_hierarchy: Node hierarchy dictionary.
            parent_name: Parent node name for building hierarchical IDs.
            
        Returns:
            List of NodeInfo objects.
        """
        layer_nodes = []
        
        for node_name, node in node_hierarchy.items():
            try:
                # Build hierarchical node ID
                full_node_id = f"{parent_name}.{node_name}" if parent_name else node_name
                
                # Extract node information
                node_info = self._extract_node_info(node, full_node_id)
                layer_nodes.append(node_info)
                
                # Recursively process children
                if hasattr(node, 'children') and node.children:
                    child_nodes = self._extract_from_hierarchy(
                        node.children, full_node_id
                    )
                    layer_nodes.extend(child_nodes)
                    
            except Exception as e:
                if self.debug:
                    print(f"Warning: Could not extract node {node_name}: {e}")
                continue
        
        return layer_nodes
    
    def _extract_node_info(self, node: Any, node_id: str) -> NodeInfo:
        """Extract information from a single node.
        
        Args:
            node: Node object from the computation graph.
            node_id: Unique identifier for the node.
            
        Returns:
            NodeInfo object containing extracted information.
        """
        node_class_name = type(node).__name__
        
        # Extract node type
        node_type = self._get_node_type(node, node_class_name)
        
        # Extract shapes
        input_shapes, output_shapes = self._extract_shapes(node, node_type)
        
        # Extract module parameters
        module_info = self._extract_module_info(node)
        
        return NodeInfo(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class_name,
            input_nodes=[],  # Will be filled in by relationship building
            output_nodes=[],  # Will be filled in by relationship building
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            weight_nodes=module_info['weight_nodes'],
            weight_shapes=module_info['weight_shapes'],
            module_args=module_info['module_args']
        )
    
    def _get_node_type(self, node: Any, node_class: str) -> str:
        """Determine the node type from the node object.
        
        Args:
            node: Node object.
            node_class: Class name of the node.
            
        Returns:
            String representing the node type.
        """
        return (
            getattr(node, 'operation', None) or
            getattr(node, 'op_name', None) or
            getattr(node, 'name', None) or
            getattr(node, '_op', None) or
            node_class.lower().replace('node', '')
        )
    
    def _extract_shapes(self, node: Any, node_type: str) -> Tuple[List[TensorShape], List[TensorShape]]:
        """Extract input and output shapes from a node.
        
        Args:
            node: Node object.
            node_type: Type of the node for special handling.
            
        Returns:
            Tuple of (input_shapes, output_shapes).
        """
        input_shapes = []
        output_shapes = []
        node_class = type(node).__name__
        
        # Special handling for TensorNode - check tensor_shape first
        if node_class == 'TensorNode':
            tensor_shape = None
            if hasattr(node, 'tensor_shape') and node.tensor_shape is not None:
                if hasattr(node.tensor_shape, '__iter__'):
                    tensor_shape = list(node.tensor_shape)
                else:
                    tensor_shape = [node.tensor_shape]
            
            if tensor_shape:
                node_name = node_type.lower() if node_type else ''
                if node_name in ['input-tensor', 'auxiliary-tensor']:
                    # Input tensors: shape is output (what they produce)
                    output_shapes = [tensor_shape]
                elif node_name == 'output-tensor':
                    # Final output tensors: shape is input (what they receive)
                    input_shapes = [tensor_shape]
                elif node_name == 'hidden-tensor':
                    # Hidden tensors: intermediate nodes have both input and output shapes
                    input_shapes = [tensor_shape]
                    output_shapes = [tensor_shape]
                else:
                    # Unknown tensor type: add to both
                    input_shapes = [tensor_shape]
                    output_shapes = [tensor_shape]
                return input_shapes, output_shapes
        
        # Extract input shapes from inputs attribute
        if hasattr(node, 'inputs') and node.inputs:
            for inp in node.inputs:
                if hasattr(inp, 'tensor_shape') and inp.tensor_shape is not None:
                    input_shapes.append(list(inp.tensor_shape))
                elif hasattr(inp, 'shape'):
                    input_shapes.append(list(inp.shape))
        elif hasattr(node, 'input_shape') and node.input_shape:
            if isinstance(node.input_shape, (list, tuple)):
                input_shapes = [list(s) for s in node.input_shape]
            else:
                input_shapes = [list(node.input_shape)]
        
        # Extract output shapes from outputs attribute
        if hasattr(node, 'outputs') and node.outputs:
            for out in node.outputs:
                if hasattr(out, 'tensor_shape') and out.tensor_shape is not None:
                    output_shapes.append(list(out.tensor_shape))
                elif hasattr(out, 'shape'):
                    output_shapes.append(list(out.shape))
        elif hasattr(node, 'output_shape') and node.output_shape:
            if isinstance(node.output_shape, (list, tuple)):
                output_shapes = [list(s) for s in node.output_shape]
            else:
                output_shapes = [list(node.output_shape)]
        
        return input_shapes, output_shapes
    
    def _extract_module_info(self, node: Any) -> Dict[str, Any]:
        """Extract module parameter information from a node.
        
        Args:
            node: Node object.
            
        Returns:
            Dictionary with weight_nodes, weight_shapes, and module_args.
        """
        module_info = {
            'weight_nodes': [],
            'weight_shapes': [],
            'module_args': {}
        }
        
        node_class = type(node).__name__
        
        if node_class == 'ModuleNode':
            self._extract_module_node_info(node, module_info)
        elif node_class == 'FunctionNode':
            self._extract_function_node_info(node, module_info)
        
        return module_info
    
    def _extract_module_node_info(self, node: Any, module_info: Dict[str, Any]) -> None:
        """Extract information from a ModuleNode.
        
        Args:
            node: ModuleNode object.
            module_info: Dictionary to populate with extracted information.
        """
        # Try to access the underlying PyTorch module
        module = self._get_pytorch_module(node)
        
        if module is not None:
            # Extract parameters
            for param_name, param in module.named_parameters(recurse=False):
                if param is not None:
                    module_info['weight_nodes'].append(param_name)
                    module_info['weight_shapes'].append(list(param.shape))
            
            # Extract buffers
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer is not None:
                    module_info['weight_nodes'].append(buffer_name)
                    module_info['weight_shapes'].append(list(buffer.shape))
            
            # Extract module arguments
            module_info['module_args'] = self._extract_module_arguments(module)
    
    def _get_pytorch_module(self, node: Any) -> Optional[nn.Module]:
        """Get the PyTorch module from a node object.
        
        Args:
            node: Node object that may contain a PyTorch module.
            
        Returns:
            PyTorch module if found, None otherwise.
        """
        # Try common attribute names
        for attr_name in MODULE_ATTR_NAMES:
            if hasattr(node, attr_name):
                attr_value = getattr(node, attr_name)
                if isinstance(attr_value, nn.Module):
                    return attr_value
        return None
    
    def _extract_module_arguments(self, module: nn.Module) -> Dict[str, Any]:
        """Extract module configuration arguments.
        
        Args:
            module: PyTorch module.
            
        Returns:
            Dictionary of module arguments.
        """
        args = {'module_type': type(module).__name__}
        
        # Extract common attributes based on module type
        for attr_name in dir(module):
            if attr_name.startswith('_') or callable(getattr(module, attr_name)):
                continue
            
            try:
                value = getattr(module, attr_name)
                
                # Handle special attribute types
                if attr_name in GEOMETRIC_ATTRS and hasattr(value, '__iter__'):
                    args[attr_name] = list(value)
                elif attr_name in BOOLEAN_ATTRS:
                    args[attr_name] = bool(value)
                elif isinstance(value, (int, float, str, bool)):
                    args[attr_name] = value
                elif attr_name == 'bias':
                    args[attr_name] = value is not None
                    
            except Exception:
                continue
        
        return args
    
    def _extract_function_node_info(self, node: Any, module_info: Dict[str, Any]) -> None:
        """Extract information from a FunctionNode.
        
        Args:
            node: FunctionNode object.
            module_info: Dictionary to populate with extracted information.
        """
        node_name = getattr(node, 'name', '').lower()
        module_info['module_args']['function_name'] = node_name
        
        # Extract from torchview 'attributes' field (contains stringified args/kwargs)
        # This is populated when collect_attributes=True in draw_graph()
        if hasattr(node, 'attributes') and node.attributes:
            parsed_args = self._parse_torchview_attributes(node.attributes, node_name)
            if parsed_args:
                module_info['module_args'].update(parsed_args)
        
        # Extract from args attribute (fallback)
        if hasattr(node, 'args') and node.args:
            for i, arg in enumerate(node.args):
                if hasattr(arg, 'shape') and hasattr(arg, 'numel'):
                    param_name = self._infer_parameter_name(node_name, i, list(arg.shape))
                    if param_name != 'input':
                        module_info['weight_nodes'].append(param_name)
                        module_info['weight_shapes'].append(list(arg.shape))
                        # Infer module arguments from parameter shapes
                        self._infer_module_arguments(node_name, param_name, list(arg.shape), module_info['module_args'])
        
        # Extract from kwargs
        if hasattr(node, 'kwargs') and node.kwargs:
            for key, value in node.kwargs.items():
                if hasattr(value, 'shape'):
                    module_info['weight_nodes'].append(key)
                    module_info['weight_shapes'].append(list(value.shape))
                else:
                    module_info['module_args'][key] = value
    
    def _parse_torchview_attributes(
        self,
        attributes: str,
        node_name: str,
    ) -> Dict[str, Any]:
        """Parse torchview stringified attributes to extract function arguments.
        
        torchview's stringify_attributes() produces strings like:
        - For functions: "[[Tensor(shape=(2, 32, 64), dtype=torch.float32), 1, 2], {}]"
        - This represents [args_list, kwargs_dict]
        
        We parse this by replacing Tensor(...) with a placeholder and using eval.
        
        Args:
            attributes: Stringified attributes from torchview.
            node_name: Name of the function (e.g., 'transpose', 'permute').
            
        Returns:
            Dictionary of parsed arguments.
        """
        result: Dict[str, Any] = {}
        
        if not attributes:
            return result
        
        # Store raw attributes for debugging
        result['raw_attributes'] = attributes
        
        try:
            # Parse the attributes string by replacing Tensor(...) with a placeholder
            args_list, kwargs_dict = self._eval_attributes_string(attributes)
            
            if args_list is None:
                return result
            
            # Extract non-tensor arguments (skip index 0 which is usually the input tensor)
            non_tensor_args = [arg for arg in args_list if not isinstance(arg, dict) or 'tensor_placeholder' not in arg]
            # Filter out tensor placeholders
            scalar_args = [arg for arg in non_tensor_args if not (isinstance(arg, dict) and 'tensor_placeholder' in arg)]
            
            if node_name == 'transpose':
                # transpose(input, dim0, dim1) - extract dim0 and dim1
                int_args = [arg for arg in scalar_args if isinstance(arg, int)]
                if len(int_args) >= 2:
                    result['dim0'] = int_args[0]
                    result['dim1'] = int_args[1]
                    result['transpose_dims'] = [int_args[0], int_args[1]]
                # Also check kwargs
                if kwargs_dict:
                    if 'dim0' in kwargs_dict:
                        result['dim0'] = kwargs_dict['dim0']
                    if 'dim1' in kwargs_dict:
                        result['dim1'] = kwargs_dict['dim1']
                    if 'dim0' in result and 'dim1' in result:
                        result['transpose_dims'] = [result['dim0'], result['dim1']]
                    
            elif node_name == 'permute':
                # permute(input, dims) or permute(input, *dims)
                int_args = [arg for arg in scalar_args if isinstance(arg, int)]
                if int_args:
                    result['permute_dims'] = int_args
                # Check for tuple/list arg
                for arg in scalar_args:
                    if isinstance(arg, (list, tuple)) and all(isinstance(d, int) for d in arg):
                        result['permute_dims'] = list(arg)
                        break
                # Check kwargs
                if kwargs_dict and 'dims' in kwargs_dict:
                    result['permute_dims'] = list(kwargs_dict['dims'])
                            
            elif node_name == 't':
                # t() is always transpose(0, 1) for 2D tensors
                result['dim0'] = 0
                result['dim1'] = 1
                result['transpose_dims'] = [1, 0]
                
            elif node_name in ('view', 'reshape'):
                # view(input, *sizes) or reshape(input, shape)
                int_args = [arg for arg in scalar_args if isinstance(arg, int)]
                if int_args:
                    result['target_shape'] = int_args
                # Check for tuple/list arg
                for arg in scalar_args:
                    if isinstance(arg, (list, tuple)) and all(isinstance(d, int) for d in arg):
                        result['target_shape'] = list(arg)
                        break
                        
        except Exception as e:
            if self.debug:
                print(f"Warning: Failed to parse attributes for {node_name}: {e}")
        
        return result
    
    def _eval_attributes_string(
        self,
        attributes: str,
    ) -> Tuple[Optional[List[Any]], Optional[Dict[str, Any]]]:
        """Safely evaluate torchview attributes string.
        
        Replaces Tensor(...) with a placeholder dict and evaluates the string.
        
        Args:
            attributes: Stringified attributes from torchview.
            
        Returns:
            Tuple of (args_list, kwargs_dict) or (None, None) on failure.
        """
        import re
        
        try:
            # Replace Tensor(shape=(...), dtype=...) with a placeholder dict
            # Pattern matches: Tensor(shape=(1, 2, 3), dtype=torch.float32)
            def replace_tensor(match: re.Match) -> str:
                return "{'tensor_placeholder': True}"
            
            # Replace all Tensor(...) occurrences
            # Handle nested parentheses by matching Tensor( then everything until matching )
            processed = attributes
            
            # Simple approach: replace Tensor(...) patterns
            # This regex handles nested parens by being greedy within the Tensor() call
            tensor_pattern = r'Tensor\([^)]*(?:\([^)]*\)[^)]*)*\)'
            processed = re.sub(tensor_pattern, "{'tensor_placeholder': True}", processed)
            
            # Replace torch.dtype references
            processed = re.sub(r'torch\.\w+', 'None', processed)
            
            # Now safely evaluate
            parsed = eval(processed, {"__builtins__": {}}, {})
            
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                args_list = parsed[0] if isinstance(parsed[0], (list, tuple)) else []
                kwargs_dict = parsed[1] if isinstance(parsed[1], dict) else {}
                return list(args_list), kwargs_dict
            
            return None, None
            
        except Exception as e:
            if self.debug:
                print(f"Warning: Failed to eval attributes: {e}")
            return None, None
    
    def _infer_parameter_name(self,
                             op_name: str,
                             arg_index: int,
                             shape: TensorShape) -> str:
        """Infer parameter name from operation type and argument position.
        
        Args:
            op_name: Operation name.
            arg_index: Position of the argument.
            shape: Shape of the tensor.
            
        Returns:
            Inferred parameter name.
        """
        op_lower = op_name.lower()
        
        if 'conv' in op_lower:
            if arg_index == 0:
                return 'input'
            elif arg_index == 1:
                return 'weight'
            elif arg_index == 2:
                return 'bias'
        elif 'linear' in op_lower:
            if arg_index == 0:
                return 'input'
            elif arg_index == 1:
                return 'weight'
            elif arg_index == 2:
                return 'bias'
        elif 'batch_norm' in op_lower or 'batchnorm' in op_lower:
            param_names = ['input', 'weight', 'bias', 'running_mean', 'running_var']
            return param_names[arg_index] if arg_index < len(param_names) else f'bn_arg_{arg_index}'
        
        return f'arg_{arg_index}' if arg_index > 0 else 'input'
    
    def _infer_module_arguments(self,
                               op_name: str,
                               param_name: str,
                               param_shape: TensorShape,
                               module_args: Dict[str, Any]) -> None:
        """Infer module configuration arguments from parameter shapes.
        
        Args:
            op_name: Operation name.
            param_name: Parameter name.
            param_shape: Shape of the parameter tensor.
            module_args: Dictionary to update with inferred arguments.
        """
        op_lower = op_name.lower()
        
        if param_name == 'weight':
            # Convolution operations
            if 'conv' in op_lower and len(param_shape) >= 3:
                if 'transpose' in op_lower:
                    # ConvTranspose weight: [in_channels, out_channels, *kernel_size]
                    module_args['in_channels'] = param_shape[0]
                    module_args['out_channels'] = param_shape[1]
                    module_args['kernel_size'] = param_shape[2:]
                else:
                    # Conv weight: [out_channels, in_channels, *kernel_size]
                    module_args['out_channels'] = param_shape[0]
                    module_args['in_channels'] = param_shape[1]
                    module_args['kernel_size'] = param_shape[2:]
            
            # Linear operations
            elif 'linear' in op_lower and len(param_shape) >= 2:
                module_args['out_features'] = param_shape[0]
                module_args['in_features'] = param_shape[1]
        
        elif param_name == 'bias':
            module_args['bias'] = True
    
    def _extract_from_edge_list(self,
                               computation_graph: Any,
                               original_model: Optional[nn.Module] = None) -> List[NodeInfo]:
        """Extract nodes from the edge_list of the computation graph.
        
        This method properly tracks node relationships (input_nodes, output_nodes)
        by processing the edge list to build connection information.
        
        Args:
            computation_graph: ComputationGraph with edge_list.
            original_model: Original PyTorch model for parameter extraction.
            
        Returns:
            List of NodeInfo objects with proper connection information.
        """
        computation_nodes = {}  # original_id -> node object
        node_order = []  # Preserve order of discovery
        
        # Step 1: Collect all unique nodes from edges
        for i, edge in enumerate(computation_graph.edge_list):
            if len(edge) >= 2:
                source_node, target_node = edge[0], edge[1]
                
                # Add source node if it's a computation node
                if self._is_computation_node(source_node):
                    original_id = str(getattr(source_node, 'node_id', id(source_node)))
                    if original_id not in computation_nodes:
                        computation_nodes[original_id] = source_node
                        node_order.append(original_id)
                
                # Add target node if it's a computation node
                if self._is_computation_node(target_node):
                    original_id = str(getattr(target_node, 'node_id', id(target_node)))
                    if original_id not in computation_nodes:
                        computation_nodes[original_id] = target_node
                        node_order.append(original_id)
        
        if self.debug:
            print(f"  Found {len(computation_nodes)} unique computation nodes")
        
        # Step 2: Generate clean IDs for all nodes and create NodeInfo objects
        result = []
        for original_id in node_order:
            node = computation_nodes[original_id]
            clean_id = self._generate_clean_id(node)
            self._original_to_clean_id[original_id] = clean_id
            
            node_info = self._extract_node_info(node, clean_id)
            result.append(node_info)
        
        # Step 3: Build relationships from edge list using the ID mapping
        id_to_node_info = {node.node_id: node for node in result}
        
        for edge in computation_graph.edge_list:
            if len(edge) >= 2:
                source_node, target_node = edge[0], edge[1]
                
                source_original_id = str(getattr(source_node, 'node_id', id(source_node)))
                target_original_id = str(getattr(target_node, 'node_id', id(target_node)))
                
                source_clean_id = self._original_to_clean_id.get(source_original_id)
                target_clean_id = self._original_to_clean_id.get(target_original_id)
                
                if source_clean_id and target_clean_id:
                    source_info = id_to_node_info.get(source_clean_id)
                    target_info = id_to_node_info.get(target_clean_id)
                    
                    if source_info and target_info:
                        # Add connection if not already present
                        if target_clean_id not in source_info.output_nodes:
                            source_info.output_nodes.append(target_clean_id)
                        if source_clean_id not in target_info.input_nodes:
                            target_info.input_nodes.append(source_clean_id)
        
        if self.debug:
            # Count nodes with connections
            nodes_with_inputs = sum(1 for n in result if n.input_nodes)
            nodes_with_outputs = sum(1 for n in result if n.output_nodes)
            print(f"  Nodes with input connections: {nodes_with_inputs}")
            print(f"  Nodes with output connections: {nodes_with_outputs}")
        
        # Step 4: Apply parameters from original model if provided
        # This handles both FunctionNode and ModuleNode cases
        if original_model:
            self._apply_model_parameters(result, original_model, computation_nodes)
        
        return result
    
    def _is_computation_node(self, node: Any) -> bool:
        """Check if a node is a computation node.
        
        Args:
            node: Node object to check.
            
        Returns:
            True if it's a computation node, False otherwise.
        """
        if not hasattr(node, '__class__'):
            return False
        node_class = type(node).__name__
        return node_class in ['TensorNode', 'ModuleNode', 'FunctionNode']
    
    def _generate_clean_id(self, node: Any) -> str:
        """Generate a clean node ID without memory addresses.
        
        Args:
            node: Node object.
            
        Returns:
            Clean node ID string.
        """
        node_name = getattr(node, 'name', type(node).__name__.lower())
        
        if node_name not in self._node_counter:
            self._node_counter[node_name] = 0
            count = 0
        else:
            self._node_counter[node_name] += 1
            count = self._node_counter[node_name]
        
        return f"Model.{node_name}_{count}" if count > 0 else f"Model.{node_name}"
    
    def _extract_from_visual_graph(self, visual_graph: Any) -> List[NodeInfo]:
        """Extract nodes from the visual graph representation.
        
        Args:
            visual_graph: graphviz.Digraph object.
            
        Returns:
            List of NodeInfo objects.
        """
        if not hasattr(visual_graph, 'source'):
            return []
        
        nodes = {}
        edges = []
        
        # Parse graphviz source
        self._parse_graphviz_source(visual_graph.source, nodes, edges)
        
        # Build relationships
        for source_id, target_id in edges:
            if source_id in nodes and target_id in nodes:
                nodes[source_id].output_nodes.append(nodes[target_id].node_id)
                nodes[target_id].input_nodes.append(nodes[source_id].node_id)
        
        return list(nodes.values())
    
    def _parse_graphviz_source(self,
                              source: str,
                              nodes: Dict[str, NodeInfo],
                              edges: List[Tuple[str, str]]) -> None:
        """Parse graphviz source to extract nodes and edges.
        
        Args:
            source: Graphviz source string.
            nodes: Dictionary to populate with nodes.
            edges: List to populate with edges.
        """
        lines = source.split('\n')
        current_node_def = ""
        in_node_definition = False
        
        for line in lines:
            line = line.strip()
            
            # Check for node definition
            if re.match(r'^\d+\s+\[label=<', line):
                in_node_definition = True
                current_node_def = line
                node_id = re.match(r'^(\d+)', line).group(1)
            elif in_node_definition and line.endswith(']'):
                current_node_def += " " + line
                in_node_definition = False
                
                # Parse node definition
                node_info = self._parse_node_definition(node_id, current_node_def)
                if node_info:
                    nodes[node_id] = node_info
                current_node_def = ""
            elif in_node_definition:
                current_node_def += " " + line
            
            # Check for edge definition
            elif '->' in line and '[' not in line:
                edge_match = re.match(r'^(\d+)\s*->\s*(\d+)', line)
                if edge_match:
                    edges.append((edge_match.group(1), edge_match.group(2)))
    
    def _parse_node_definition(self, node_id: str, node_def: str) -> Optional[NodeInfo]:
        """Parse a node definition from graphviz source.
        
        Args:
            node_id: Node ID from graphviz.
            node_def: Node definition string.
            
        Returns:
            NodeInfo object if successfully parsed, None otherwise.
        """
        try:
            # Extract node type
            node_type = "Unknown"
            type_match = re.search(r'<TD[^>]*>([^<]*)<BR/>depth:\d+</TD>', node_def)
            if type_match:
                node_type = type_match.group(1).strip()
            
            # Extract shapes
            input_shapes = []
            output_shapes = []
            
            if node_type in ["input-tensor", "output-tensor"]:
                shape_match = re.search(r'<TD>\(([^)]+)\)</TD>', node_def)
                if shape_match:
                    shape_str = shape_match.group(1)
                    shape = [int(x.strip()) for x in shape_str.split(',')]
                    if node_type == "input-tensor":
                        output_shapes.append(shape)
                    else:
                        input_shapes.append(shape)
            
            # Create node ID
            hierarchical_id = f"Model.{node_type}_{node_id}"
            
            return NodeInfo(
                node_id=hierarchical_id,
                node_type=node_type,
                input_shapes=input_shapes,
                output_shapes=output_shapes
            )
            
        except Exception as e:
            if self.debug:
                print(f"Error parsing node {node_id}: {e}")
            return None
    
    def _apply_model_parameters(self,
                               layer_nodes: List[NodeInfo],
                               model: nn.Module,
                               computation_nodes: Optional[Dict[str, Any]] = None) -> None:
        """Apply parameters from the original model to extracted nodes.
        
        This uses shape-based matching to correctly associate PyTorch modules
        with their corresponding function nodes. Also handles ModuleNode cases
        where the node directly references a PyTorch module.
        
        Args:
            layer_nodes: List of NodeInfo objects to update.
            model: Original PyTorch model.
            computation_nodes: Optional dict mapping original IDs to node objects.
        """
        # First, try to extract from ModuleNode objects directly
        if computation_nodes:
            for original_id, node in computation_nodes.items():
                if type(node).__name__ == 'ModuleNode':
                    clean_id = self._original_to_clean_id.get(original_id)
                    if clean_id:
                        node_info = next((n for n in layer_nodes if n.node_id == clean_id), None)
                        if node_info and node_info.node_id not in self._processed_nodes:
                            pytorch_module = self._get_pytorch_module(node)
                            if pytorch_module:
                                module_type = type(pytorch_module).__name__
                                self._apply_module_to_node(node_info, pytorch_module, module_type)
                                self._processed_nodes.add(node_info.node_id)
                                if self.debug:
                                    print(f"  Applied module from ModuleNode: {node_info.node_id}")
        
        # Collect all modules from the model
        modules_by_type: Dict[str, List[Tuple[str, nn.Module]]] = {}
        for name, module in model.named_modules():
            if name == '':  # Skip root
                continue
            module_type = type(module).__name__
            if module_type not in modules_by_type:
                modules_by_type[module_type] = []
            modules_by_type[module_type].append((name, module))
        
        if self.debug:
            print(f"  Found {sum(len(v) for v in modules_by_type.values())} PyTorch modules")
        
        # Match modules to nodes by type and shape
        for module_type, modules_list in modules_by_type.items():
            # Find candidate nodes for this module type (both FunctionNode and ModuleNode)
            candidate_nodes = [
                node for node in layer_nodes
                if node.node_class in ('FunctionNode', 'ModuleNode')
                and module_type.lower() in node.node_type.lower()
                and node.node_id not in self._processed_nodes
            ]
            
            if self.debug and modules_list:
                print(f"  Matching {len(modules_list)} {module_type} modules to {len(candidate_nodes)} nodes")
            
            # Special handling for Linear layers with shape matching
            if module_type == 'Linear' and len(modules_list) > len(candidate_nodes) > 0:
                self._match_linear_modules_by_shape(modules_list, candidate_nodes)
            else:
                # Standard sequential matching
                for i, (module_name, module) in enumerate(modules_list):
                    if module_name in self._matched_modules:
                        continue
                    
                    target_node = None
                    for node in candidate_nodes:
                        if node.node_id not in self._processed_nodes:
                            target_node = node
                            break
                    
                    if target_node:
                        self._apply_module_to_node(target_node, module, module_type)
                        self._processed_nodes.add(target_node.node_id)
                        self._matched_modules.add(module_name)
    
    def _match_linear_modules_by_shape(self,
                                       modules_list: List[Tuple[str, nn.Module]],
                                       candidate_nodes: List[NodeInfo]) -> None:
        """Match Linear modules to nodes using shape-based matching.
        
        Args:
            modules_list: List of (name, module) tuples.
            candidate_nodes: List of candidate NodeInfo objects.
        """
        for node in candidate_nodes:
            if node.node_id in self._processed_nodes:
                continue
            
            # Get expected dimensions from node shapes
            if node.input_shapes and node.output_shapes:
                input_shape = node.input_shapes[0]
                output_shape = node.output_shapes[0]
                
                if len(input_shape) > 0 and len(output_shape) > 0:
                    expected_in_features = input_shape[-1]
                    expected_out_features = output_shape[-1]
                    
                    # Find matching Linear module
                    for module_name, module in modules_list:
                        if module_name in self._matched_modules:
                            continue
                        
                        if (hasattr(module, 'in_features') and 
                            hasattr(module, 'out_features')):
                            if (module.in_features == expected_in_features and
                                module.out_features == expected_out_features):
                                self._apply_module_to_node(node, module, 'Linear')
                                self._processed_nodes.add(node.node_id)
                                self._matched_modules.add(module_name)
                                if self.debug:
                                    print(f"    Shape match: {node.node_id} <-> {module_name}")
                                break
    
    def _apply_module_to_node(self,
                            node: NodeInfo,
                            module: nn.Module,
                            module_type: str) -> None:
        """Apply module parameters to a node.
        
        Args:
            node: NodeInfo to update.
            module: PyTorch module with parameters.
            module_type: Type name of the module.
        """
        # Clear existing parameters to avoid duplicates
        node.weight_nodes.clear()
        node.weight_shapes.clear()
        
        # Extract parameters
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None:
                node.weight_nodes.append(param_name)
                node.weight_shapes.append(list(param.shape))
        
        # Extract buffers
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                node.weight_nodes.append(buffer_name)
                node.weight_shapes.append(list(buffer.shape))
        
        # Update module arguments
        node.module_args['module_type'] = module_type
        node.module_args.update(self._extract_module_arguments(module))
    
    def _save_pytorch_graph_yaml(
        self,
        layer_nodes: List[NodeInfo],
        filename: Path,
        *,
        model_name: str,
    ) -> None:
        """Save extracted nodes to a structured YAML graph.

        The YAML format matches the original process_torchview_graph.py output:

          model_name: <str>
          layers:
            <node_id>:
              type: <str>
              node_class: <str>
              input_shapes: [...]
              output_shapes: [...]
              weight_nodes: [...]
              weight_shapes: [...]
              module_args: {...}
              connections:
                inputs: [...]
                outputs: [...]

        Args:
            layer_nodes: Extracted nodes.
            filename: Output YAML path.
            model_name: Human-readable model name.
        """
        graph_dict: Dict[str, Any] = {
            "model_name": model_name,
            "layers": {},
        }

        for node in layer_nodes:
            graph_dict["layers"][node.node_id] = {
                "type": node.node_type,
                "node_class": node.node_class,
                "input_shapes": node.input_shapes,
                "output_shapes": node.output_shapes,
                "weight_nodes": node.weight_nodes,
                "weight_shapes": node.weight_shapes,
                "module_args": node.module_args,
                "connections": {
                    "inputs": node.input_nodes,
                    "outputs": node.output_nodes,
                },
            }

        with open(filename, "w") as f:
            from solar.common.utils import NoAliasDumper
            yaml.dump(graph_dict, f, Dumper=NoAliasDumper, sort_keys=False, default_flow_style=False)

        if self.debug:
            print(f"PyTorch graph YAML saved to {filename}")
    
    def _print_layer_summary(self, layer_nodes: List[NodeInfo]) -> None:
        """Print summary of extracted layer nodes.
        
        Args:
            layer_nodes: List of NodeInfo objects.
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTED LAYER NODES ({len(layer_nodes)} nodes)")
        print(f"{'='*80}")
        
        for i, node in enumerate(layer_nodes[:5], 1):  # Show first 5
            print(f"\n[{i}] Node ID: {node.node_id}")
            print(f"    Type: {node.node_type} ({node.node_class})")
            print(f"    Input Nodes: {node.input_nodes}")
            print(f"    Output Nodes: {node.output_nodes}")
            print(f"    Input Shapes: {node.input_shapes}")
            print(f"    Output Shapes: {node.output_shapes}")
            if node.weight_nodes:
                print(f"    Weights: {node.weight_nodes}")
        
        if len(layer_nodes) > 5:
            print(f"\n... and {len(layer_nodes) - 5} more nodes")
