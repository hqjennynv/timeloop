"""Common utility functions for the Solar package.

This module provides utility functions following Google's Python style guide.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from solar.common.constants import SAFE_ENV_VARS


class NoAliasDumper(yaml.SafeDumper):
    """YAML dumper that disables anchors/aliases for human-readable output.
    
    Standard PyYAML creates anchors (&id001) and aliases (*id001) when it
    detects duplicate data structures. This dumper disables that behavior
    to produce more readable YAML files, at the cost of slightly larger
    file sizes.
    
    Example:
        Instead of:
            input_shapes: &id001
            - - 16
            output_shapes: *id001
        
        This produces:
            input_shapes:
            - - 16
            output_shapes:
            - - 16
    """
    def ignore_aliases(self, data):
        return True


def format_number(n: int) -> str:
    """Format a number with magnitude suffix for readability.
    
    Args:
        n: Number to format.
        
    Returns:
        Formatted string with appropriate suffix (K, M, B, T).
        
    Examples:
        >>> format_number(1500)
        '1.50K'
        >>> format_number(2500000)
        '2.50M'
    """
    if n < 1000:
        return str(n)
    elif n < 1_000_000:
        return f"{n/1000:.2f}K"
    elif n < 1_000_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n < 1_000_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    else:
        return f"{n/1_000_000_000_000:.2f}T"


def setup_safe_environment() -> None:
    """Set up environment variables for safe execution.
    
    This function configures environment variables to prevent
    segfaults and OOM issues during model processing.
    """
    for var, value in SAFE_ENV_VARS.items():
        os.environ[var] = value
    print("🔒 Safe environment configured: single-threaded, CPU-only")


def load_module_from_file(file_path: Union[str, Path]) -> Any:
    """Dynamically load a Python module from a file path.
    
    Args:
        file_path: Path to the Python file to load.
        
    Returns:
        The loaded module object.
        
    Raises:
        ImportError: If the module cannot be loaded.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory.
        
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_prefix(filename: str, is_cudacoder: bool = False) -> str:
    """Extract the prefix from a filename.
    
    Args:
        filename: The filename (e.g., "1_Square_matrix_multiplication_.py" or "100.py").
        is_cudacoder: Whether this is a cudacoder file.
        
    Returns:
        The prefix (e.g., "1" or "100").
    """
    base_name = Path(filename).stem
    if is_cudacoder:
        # For cudacoder, return normalized numeric ID
        try:
            return str(int(base_name))
        except ValueError:
            return base_name
    else:
        # For kernelbench, extract everything before first underscore
        return base_name.split('_')[0]


def parse_kernel_ids(kernel_ids: Optional[List[int]], 
                    available_files: List[Path]) -> List[Path]:
    """Filter files by kernel IDs.
    
    Args:
        kernel_ids: List of kernel IDs to filter by.
        available_files: List of available file paths.
        
    Returns:
        Filtered list of file paths matching the kernel IDs.
    """
    if kernel_ids is None:
        return available_files
    
    kernel_ids_str = [str(kid) for kid in kernel_ids]
    filtered_files = []
    
    for file_path in available_files:
        prefix = get_file_prefix(file_path.name)
        if prefix in kernel_ids_str:
            filtered_files.append(file_path)
    
    return filtered_files


def merge_dicts(base: Dict[str, Any], 
                update: Dict[str, Any],
                deep: bool = True) -> Dict[str, Any]:
    """Merge two dictionaries.
    
    Args:
        base: Base dictionary to merge into.
        update: Dictionary with updates to apply.
        deep: Whether to perform deep merge for nested dicts.
        
    Returns:
        Merged dictionary (modifies base in-place and returns it).
    """
    for key, value in update.items():
        if deep and key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_dicts(base[key], value, deep=True)
        else:
            base[key] = value
    return base


def validate_shapes(shapes: Dict[str, List[int]]) -> bool:
    """Validate tensor shapes dictionary.
    
    Args:
        shapes: Dictionary mapping tensor names to shapes.
        
    Returns:
        True if all shapes are valid, False otherwise.
    """
    if not shapes:
        return False
    
    for name, shape in shapes.items():
        if not isinstance(shape, (list, tuple)):
            return False
        if not shape:  # Empty shape
            return False
        if not all(isinstance(dim, int) and dim > 0 for dim in shape):
            return False
    
    return True


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert (can be nested).
        
    Returns:
        Object with numpy types converted to Python types.
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(item) for item in obj)
    else:
        return obj


def parse_dim_tokens(dims_str: str) -> List[str]:
    """Parse a dimension string into individual dimension tokens.
    
    Tokens are in the format: single capital letter optionally followed by digit(s).
    Examples: A, B, A1, B1, A2, Z99, etc.
    
    All tokens are returned in uppercase for consistency.
    
    Examples:
        "ABC" -> ["A", "B", "C"]
        "A1B1C1" -> ["A1", "B1", "C1"]
        "A1B2C3" -> ["A1", "B2", "C3"]
        "ABCA1B1" -> ["A", "B", "C", "A1", "B1"]
        "A12B34" -> ["A12", "B34"]
        
    Args:
        dims_str: String of dimension names (e.g., "ABC", "A1B1C1", "A12B34")
        
    Returns:
        List of individual dimension tokens (uppercase)
    """
    if not dims_str:
        return []
    
    tokens = []
    i = 0
    while i < len(dims_str):
        if not dims_str[i].isalpha():
            # Skip non-alphabetic characters
            i += 1
            continue
        
        # Get the single letter
        letter = dims_str[i].upper()
        i += 1
        
        # Check if followed by digits
        if i < len(dims_str) and dims_str[i].isdigit():
            # Collect all following digits
            j = i
            while j < len(dims_str) and dims_str[j].isdigit():
                j += 1
            digits = dims_str[i:j]
            tokens.append(letter + digits)
            i = j
        else:
            # No digits following - just the single letter
            tokens.append(letter)
    
    return tokens


def parse_einsum_equation(equation: str) -> tuple:
    """Parse an einsum equation into input operand tokens and output tokens.
    
    Tokens are in the format: single capital letter optionally followed by digit(s).
    Examples: A, B, A1, B1, A2, Z99, etc.
    
    Examples:
        "ABC,DE->ADE" -> ([["A", "B", "C"], ["D", "E"]], ["A", "D", "E"])
        "A1B1C1,D1E1->A1D1E1" -> ([["A1", "B1", "C1"], ["D1", "E1"]], ["A1", "D1", "E1"])
        "->ABC" -> ([], ["A", "B", "C"])  # start node
        "->A1B1C1" -> ([], ["A1", "B1", "C1"])  # start node with numbered dims
        
    Args:
        equation: Einsum equation string
        
    Returns:
        Tuple of (list of input operand token lists, output tokens)
    """
    if not equation or "->" not in equation:
        return [], []
    
    parts = equation.split("->")
    if len(parts) != 2:
        return [], []
    
    lhs, rhs = parts[0].strip(), parts[1].strip()
    
    # Parse output tokens
    output_tokens = parse_dim_tokens(rhs)
    
    # Parse input operands (comma-separated)
    input_operands: List[List[str]] = []
    if lhs:
        for operand_str in lhs.split(","):
            operand_str = operand_str.strip()
            if operand_str:
                input_operands.append(parse_dim_tokens(operand_str))
    
    return input_operands, output_tokens


def load_einsum_graph_to_networkx(layers: Dict[str, Any]) -> Any:
    """Build a NetworkX DiGraph from einsum graph layers dict.
    
    Each node is a layer (operation), and edges represent data flow
    based on the connections.inputs and connections.outputs fields.
    
    Args:
        layers: Dictionary of layer_id -> layer_data from einsum_graph.yaml
        
    Returns:
        NetworkX DiGraph with nodes and edges representing the computation graph
    """
    import networkx as nx
    
    graph = nx.DiGraph()
    
    # Add all nodes with their data
    for layer_id, layer_data in layers.items():
        graph.add_node(layer_id, **layer_data)
    
    # Add edges based on connections
    for layer_id, layer_data in layers.items():
        connections = layer_data.get('connections', {})
        
        # Add edges from input nodes to this node
        for input_id in connections.get('inputs', []):
            if input_id in layers:
                graph.add_edge(input_id, layer_id)
        
        # Add edges from this node to output nodes
        for output_id in connections.get('outputs', []):
            if output_id in layers:
                graph.add_edge(layer_id, output_id)
    
    return graph
