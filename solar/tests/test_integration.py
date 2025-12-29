"""Integration tests for Solar package with kernelbench and cudacoder."""

import pytest
import json
import yaml
from pathlib import Path
import tempfile
import shutil

from solar.graph import BenchmarkProcessor
from solar.einsum import EinsumAnalyzer, PyTorchToEinsum
from solar.common.types import ProcessingConfig


def create_sample_kernelbench_model(path: Path) -> None:
    """Create a sample kernelbench model file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 56 * 56, 1000)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

def get_inputs():
    return [torch.randn(1, 3, 224, 224)]
""")


def create_sample_cudacoder_model(path: Path) -> None:
    """Create a sample cudacoder model file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = F.log_softmax(self.linear3(x), dim=1)
        return x

def get_inputs():
    return [torch.randn(32, 784)]
""")


class TestKernelbenchIntegration:
    """Integration tests for kernelbench models."""
    
    def test_full_kernelbench_pipeline(self):
        """Test complete pipeline for kernelbench model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample model
            model_dir = Path(tmpdir) / "kernelbench" / "level1"
            model_path = model_dir / "1_test_model.py"
            create_sample_kernelbench_model(model_path)
            
            # Process the model
            output_dir = Path(tmpdir) / "outputs"
            config = ProcessingConfig(
                output_dir=str(output_dir),
                save_graph=False,
                force_rerun=True,
                debug=False
            )
            processor = BenchmarkProcessor(config)
            
            success = processor.process_file(
                str(model_path),
                is_cudacoder=False
            )
            
            assert success is True
            
            # Check output files
            kernel_output = output_dir / "level1" / "1"
            assert kernel_output.exists()
            
            # Check extracted layer nodes
            pytorch_graph = kernel_output / "pytorch_graph.yaml"
            assert pytorch_graph.exists()

            # PyTorch graph -> einsum graph (also generates einsum_graph_renamed.yaml).
            converter = PyTorchToEinsum(debug=False, enable_agent=False)
            einsum_graph = converter.convert_graph(pytorch_graph, kernel_output)
            assert einsum_graph is not None
            assert (kernel_output / "einsum_graph.yaml").exists()
            assert (kernel_output / "einsum_graph_renamed.yaml").exists()
    
    def test_kernelbench_batch_processing(self):
        """Test processing multiple kernelbench models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple models
            model_dir = Path(tmpdir) / "kernelbench" / "level1"
            for i in range(1, 4):
                model_path = model_dir / f"{i}_model.py"
                create_sample_kernelbench_model(model_path)
            
            # Process all models
            output_dir = Path(tmpdir) / "outputs"
            config = ProcessingConfig(
                output_dir=str(output_dir),
                save_graph=False,
                force_rerun=True,
                debug=False
            )
            processor = BenchmarkProcessor(config)
            
            results = processor.process_directory(
                str(tmpdir),
                level="level1",
                kernel_ids=None,  # Process all
                is_cudacoder=False
            )
            
            # Check all were processed
            assert len(results) == 3
            assert all(results.values())
            
            # Check output directories
            for i in range(1, 4):
                kernel_output = output_dir / "level1" / str(i)
                assert kernel_output.exists()


class TestCudacoderIntegration:
    """Integration tests for cudacoder models."""
    
    def test_full_cudacoder_pipeline(self):
        """Test complete pipeline for cudacoder model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample model
            model_dir = Path(tmpdir) / "cudacoder" / "level0"
            model_path = model_dir / "001.py"
            create_sample_cudacoder_model(model_path)
            
            # Process the model
            output_dir = Path(tmpdir) / "outputs"
            config = ProcessingConfig(
                output_dir=str(output_dir),
                save_graph=False,
                force_rerun=True,
                debug=False
            )
            processor = BenchmarkProcessor(config)
            
            success = processor.process_file(
                str(model_path),
                is_cudacoder=True
            )
            
            assert success is True
            
            # Check output files
            kernel_output = output_dir / "level0" / "1"
            assert kernel_output.exists()
            
            # Check extracted layer nodes
            pytorch_graph = kernel_output / "pytorch_graph.yaml"
            assert pytorch_graph.exists()

            # PyTorch graph -> einsum graph (also generates einsum_graph_renamed.yaml).
            converter = PyTorchToEinsum(debug=False, enable_agent=False)
            einsum_graph = converter.convert_graph(pytorch_graph, kernel_output)
            assert einsum_graph is not None
            assert (kernel_output / "einsum_graph.yaml").exists()
            assert (kernel_output / "einsum_graph_renamed.yaml").exists()
    
    def test_cudacoder_numeric_ids(self):
        """Test handling of cudacoder numeric IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create models with various numeric IDs
            model_dir = Path(tmpdir) / "cudacoder" / "level0"
            test_ids = ["001.py", "010.py", "100.py", "999.py"]
            
            for filename in test_ids:
                model_path = model_dir / filename
                create_sample_cudacoder_model(model_path)
            
            # Process specific IDs
            output_dir = Path(tmpdir) / "outputs"
            config = ProcessingConfig(
                output_dir=str(output_dir),
                save_graph=False,
                force_rerun=True,
                debug=False
            )
            processor = BenchmarkProcessor(config)
            
            results = processor.process_directory(
                str(tmpdir),
                level="level0",
                kernel_ids=[1, 100],  # Should match 001.py and 100.py
                is_cudacoder=True
            )
            
            assert len(results) == 2
            
            # Check correct output directories
            assert (output_dir / "level0" / "1").exists()
            assert (output_dir / "level0" / "100").exists()
            assert not (output_dir / "level0" / "10").exists()


class TestCrossCompatibility:
    """Test cross-compatibility between kernelbench and cudacoder."""
    
    def test_mixed_processing(self):
        """Test processing both kernelbench and cudacoder models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create both types of models
            kb_path = Path(tmpdir) / "kernelbench" / "level1" / "1_model.py"
            cc_path = Path(tmpdir) / "cudacoder" / "level0" / "001.py"
            
            create_sample_kernelbench_model(kb_path)
            create_sample_cudacoder_model(cc_path)
            
            # Process both
            output_dir = Path(tmpdir) / "outputs"
            config = ProcessingConfig(
                output_dir=str(output_dir),
                save_graph=False,
                force_rerun=True,
                debug=False
            )
            processor = BenchmarkProcessor(config)
            
            # Process kernelbench
            kb_success = processor.process_file(str(kb_path), is_cudacoder=False)
            assert kb_success is True
            
            # Process cudacoder
            cc_success = processor.process_file(str(cc_path), is_cudacoder=True)
            assert cc_success is True

            converter = PyTorchToEinsum(debug=False, enable_agent=False)

            kb_graph = output_dir / "level1" / "1" / "pytorch_graph.yaml"
            cc_graph = output_dir / "level0" / "1" / "pytorch_graph.yaml"

            kb_einsum = converter.convert_graph(kb_graph, output_dir / "level1" / "1")
            cc_einsum = converter.convert_graph(cc_graph, output_dir / "level0" / "1")
            assert kb_einsum is not None
            assert cc_einsum is not None
    
    def test_unified_einsum_analysis(self):
        """Test einsum analyzer works with both naming conventions."""
        analyzer = EinsumAnalyzer()
        
        # Kernelbench-style names
        kb_names = ["Conv2d", "Linear", "BatchNorm2d", "ReLU"]
        for name in kb_names:
            normalized = analyzer._get_operation_from_name(name)
            assert normalized is not None
        
        # Cudacoder-style names
        cc_names = ["conv2d", "linear", "batch_norm", "relu"]
        for name in cc_names:
            normalized = analyzer._get_operation_from_name(name)
            assert normalized is not None
        
        # Both should map to same operations
        assert analyzer._get_operation_from_name("Conv2d") == analyzer._get_operation_from_name("conv2d")
        assert analyzer._get_operation_from_name("Linear") == analyzer._get_operation_from_name("linear")
