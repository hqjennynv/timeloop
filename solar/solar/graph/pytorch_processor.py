"""PyTorch model processor for extracting and analyzing PyTorch models.

This module processes PyTorch model files (including kernelbench and cudacoder) 
to extract computation graphs and model information.
"""

import gc
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torchview
from torch import nn

from solar.common.types import ProcessingConfig
from solar.common.utils import (
    ensure_directory,
    load_module_from_file,
    setup_safe_environment,
)
from solar.graph.torchview_processor import TorchviewProcessor


class PyTorchProcessor:
    """Process PyTorch model into a saved torch graph."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the PyTorchProcessor.
        
        Args:
            config: Processing configuration. If None, uses defaults.
        """
        self.config = config or ProcessingConfig()
        self.torchview_processor = TorchviewProcessor(debug=self.config.debug)
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Set up safe execution environment."""
        if self.config.safe_mode:
            setup_safe_environment()
    
    def process_model_file(self, file_path: str, output_dir: str) -> bool:
        """Process PyTorch model file into `output_dir`.

        Args:
            file_path: Path to a Python file containing `Model` and `get_inputs()`.
            output_dir: Output directory to write graph artifacts to.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.config.debug:
                print(f"Processing {file_path}...")

            output_path = Path(output_dir)
            ensure_directory(output_path)

            # Check if already processed
            if self._is_already_processed(output_path) and not self.config.force_rerun:
                print(f"Skipping {file_path} - already processed")
                return True

            # Copy source file for reproducibility (best effort).
            try:
                source_copy = output_path / f"source_{Path(file_path).name}"
                if not source_copy.exists():
                    import shutil

                    shutil.copy2(file_path, source_copy)
            except Exception:
                pass

            # Load and process model
            model, inputs = self._load_model(file_path)
            if model is None:
                return False
            
            # Generate torchview graph (pass output_path for saving visualization)
            graph = self._generate_torchview_graph(model, inputs, str(output_path))
            if graph is None:
                return False
            
            # Process and save graph
            kernel_name = Path(file_path).stem
            self.torchview_processor.process_graph(
                graph, str(output_path), kernel_name, original_model=model
            )
            
            # Clean up
            self._cleanup(model, inputs, graph)
            
            print(f"✓ Successfully processed {file_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def _is_already_processed(self, output_dir: Path) -> bool:
        """Check if a model has already been processed.
        
        Args:
            output_dir: Output directory to check.
            
        Returns:
            True if already processed, False otherwise.
        """
        # Canonical artifact.
        return (output_dir / "pytorch_graph.yaml").exists()
    
    def _load_model(self,
                   file_path: str) -> Tuple[Optional[nn.Module], Optional[Any]]:
        """Load a model from a Python file.
        
        Args:
            file_path: Path to the model file.
            
        Returns:
            Tuple of (model, inputs) or (None, None) if failed.
        """
        try:
            module = load_module_from_file(file_path)
            
            # Check for required components
            if not hasattr(module, 'Model'):
                print(f"Warning: No Model class found in {file_path}")
                return None, None
            
            if not hasattr(module, 'get_inputs'):
                print(f"Warning: No get_inputs function found in {file_path}")
                return None, None
            
            # Create model instance
            if hasattr(module, 'get_init_inputs'):
                try:
                    init_inputs = module.get_init_inputs()
                    model = module.Model(*init_inputs) if init_inputs else module.Model()
                except Exception:
                    model = module.Model()
            else:
                model = module.Model()
            
            # Get inputs
            inputs = module.get_inputs()
            
            return model, inputs
            
        except Exception as e:
            print(f"Error loading model from {file_path}: {e}")
            return None, None
    
    def _generate_torchview_graph(
        self,
        model: nn.Module,
        inputs: Any,
        output_dir: Optional[str] = None,
    ) -> Optional[Any]:
        """Generate a torchview computation graph.
        
        Args:
            model: PyTorch model.
            inputs: Model inputs.
            output_dir: Directory to save graph visualization (if save_graph enabled).
            
        Returns:
            Computation graph or None if failed.
        """
        devices_to_try = ["meta", "cpu"]
        
        for device in devices_to_try:
            try:
                # Prepare model
                model = model.to_empty(device=device)
                model.eval()
                
                # Check for RNN-like models that need CPU
                if device == "meta" and self._is_rnn_model(model):
                    continue
                
                # Generate graph (don't let torchview save - we'll do it ourselves)
                graph = torchview.draw_graph(
                    model,
                    input_data=inputs,
                    device=device,
                    save_graph=False,  # We handle saving separately
                    expand_nested=True,
                    depth=float('inf'),
                    hide_module_functions=False,
                    hide_inner_tensors=False,
                    roll=False,
                    strict=False,
                    collect_attributes=True,  # Capture function/module args
                )
                
                if self.config.debug:
                    print(f"✅ Generated graph using {device} device")
                
                # Save graph visualization if requested
                if self.config.save_graph and output_dir:
                    self._save_torchview_graph(graph, output_dir)
                
                return graph
                
            except (NotImplementedError, RuntimeError) as e:
                if device == "meta":
                    if self.config.debug:
                        print(f"⚠️ Meta device failed: {e}")
                    continue
                else:
                    raise
            except Exception as e:
                if device == "meta":
                    continue
                raise
        
        return None

    def _save_torchview_graph(self, graph: Any, output_dir: str) -> None:
        """Save torchview graph visualization to the output directory.
        
        Args:
            graph: Torchview graph object.
            output_dir: Directory to save the graph visualization.
        """
        try:
            output_path = Path(output_dir)
            graph_filename = output_path / "torchview_graph"
            
            # torchview's visual_graph is a graphviz.Digraph object
            if hasattr(graph, 'visual_graph'):
                # Render to PDF (default) and PNG
                graph.visual_graph.render(
                    filename=str(graph_filename),
                    format='pdf',
                    cleanup=True,  # Remove the intermediate .gv file
                )
                if self.config.debug:
                    print(f"📊 Saved torchview graph: {graph_filename}.pdf")
            else:
                if self.config.debug:
                    print("⚠️ Graph object does not have visual_graph attribute")
        except Exception as e:
            if self.config.debug:
                print(f"⚠️ Failed to save torchview graph: {e}")
    
    def _is_rnn_model(self, model: nn.Module) -> bool:
        """Check if a model is RNN-like.
        
        Args:
            model: PyTorch model.
            
        Returns:
            True if RNN-like, False otherwise.
        """
        # Check for RNN attributes
        if hasattr(model, 'hidden'):
            return True
        
        # Check module names
        for name, _ in model.named_modules():
            if any(rnn_type in name.lower() for rnn_type in ['rnn', 'gru', 'lstm']):
                return True
        
        return False
    
    def _cleanup(self, *objects: Any) -> None:
        """Clean up objects and run garbage collection.
        
        Args:
            *objects: Objects to delete.
        """
        for obj in objects:
            del obj
        gc.collect()
