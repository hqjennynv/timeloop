# Solar: PyTorch Model Einsum Toolkit

Solar is a toolkit for extracting PyTorch model graphs and converting them to einsum representations.

## Features

- **Graph Extraction**: Extract structured computation graphs from PyTorch models (torchview-based)
- **Einsum Conversion**: Convert PyTorch operations to einsum notation with automatic rank renaming
- **Graph Visualization**: Generate PDF visualizations of einsum graphs
- **Timeloop Export**: Convert to Timeloop workload format for architectural exploration
- **Human-Readable YAML**: All outputs use clean YAML without anchors/aliases

## Installation

```bash
# Install Solar in development mode
cd solar
pip install -e .
```

Dependencies:
```bash
# Core dependencies are in requirements.txt
pip install -r requirements.txt

# For graph visualization (optional)
pip install graphviz matplotlib
```

## The Pipeline

Solar processes models through three stages:

```
Stage 1: PyTorch Graph Extraction
  └─> pytorch_graph.yaml

Stage 2: Einsum Conversion + Rank Renaming
  └─> einsum_graph.yaml
  └─> einsum_graph_renamed.yaml
  └─> einsum_graph.pdf (optional)

Stage 3: Timeloop Export (optional)
  └─> timeloop_graph.yaml
```

## Examples

Solar includes several example models demonstrating different attention patterns:

### Available Examples

| Example | Description |
|---------|-------------|
| `Attention/` | Multi-head self-attention |
| `BERT/` | BERT-like encoder model |
| `DenseAttention/` | Full attention matrix |
| `SlidingWindowAttention/` | Local window attention |
| `RandomAttention/` | Random sparse attention |
| `BlockSparseAttention/` | Block-sparse attention |

### Running an Example

```bash
# Run the complete pipeline for any example
cd solar/examples/DenseAttention
bash run_solar.sh

# Outputs:
#   - output/graph/pytorch_graph.yaml           (Stage 1)
#   - output/einsum/einsum_graph.yaml           (Stage 2)
#   - output/einsum/einsum_graph_renamed.yaml   (Stage 2 - with BFS rank renaming)
#   - output/einsum/einsum_graph.pdf            (Stage 2 - visualization)
#   - output/timeloop/timeloop_graph.yaml       (Stage 3)
```

## CLI Commands

### Single Model Processing

```bash
# Stage 1: Extract PyTorch graph
solar-process-model --model-file model.py --output-dir output/graph

# Stage 2: Convert to einsum (with optional PDF visualization)
solar-toeinsum-model --graph-path output/graph/pytorch_graph.yaml \
                     --output-dir output/einsum --no-copy-graph \
                     --save-graph

# Stage 3: Convert to Timeloop format
solar-totimeloop --einsum-graph-path output/einsum/einsum_graph.yaml \
                 --output-dir output/timeloop
```

### Benchmark Processing

```bash
# All-in-one: process + convert to einsum
solar-toeinsum --level level1 --kernel-ids 1 2 3

# Show kernel status
solar-toeinsum --kernel-status --level level1

# Collect operation statistics
solar-toeinsum --level level1 --collect-stats

# List available analyses
solar-toeinsum --list-analyses
```

## Output File Formats

All output files use **human-readable YAML** without anchors/aliases:

- **pytorch_graph.yaml**: Structured graph with layers, shapes, weights, connections
- **einsum_graph.yaml**: Einsum equations + shapes for each layer
- **einsum_graph_renamed.yaml**: Einsum graph with consistent dimension labels (BFS-based)
- **einsum_graph.pdf**: Visual representation of the computation graph
- **timeloop_graph.yaml**: Timeloop workload format for architectural exploration

## Testing

```bash
# Run all tests
bash run_tests.sh

# Quick smoke tests
bash run_tests.sh quick

# Run specific test categories
bash run_tests.sh graph      # Graph processing tests
bash run_tests.sh einsum     # Einsum analyzer tests
bash run_tests.sh unit       # All unit tests
bash run_tests.sh integration # Integration tests

# Test examples
bash run_tests.sh examples   # Run all example scripts

# Verbose output
bash run_tests.sh all -v
```

See `TESTING_GUIDE.md` for detailed testing documentation.

## Python API

### Graph Processing

```python
from solar.graph import PyTorchProcessor

processor = PyTorchProcessor()
processor.process_model_file("model.py", output_dir="outputs/my_model")
```

### Einsum Conversion

```python
from solar.einsum import PyTorchToEinsum

converter = PyTorchToEinsum()
einsum_graph = converter.convert(
    "outputs/my_model/pytorch_graph.yaml",
    "outputs/my_model",
    copy_graph=False  # Don't duplicate input graph
)
# Produces both einsum_graph.yaml and einsum_graph_renamed.yaml
```

### Graph Visualization

```python
from solar.einsum import EinsumGraphVisualizer

visualizer = EinsumGraphVisualizer()
visualizer.save_graph_pdf(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model/einsum_graph.pdf"
)
```

### Timeloop Export

```python
from solar.einsum import EinsumToTimeloop

converter = EinsumToTimeloop()
result = converter.convert(
    "outputs/my_model/einsum_graph_renamed.yaml",
    "outputs/my_model/timeloop_graph.yaml"
)
```

## Architecture

```
solar/
├── solar/
│   ├── common/        # Shared types, constants, utilities (NoAliasDumper)
│   ├── graph/         # Stage 1: PyTorch graph extraction
│   ├── einsum/        # Stage 2: Einsum conversion, visualization, Timeloop export
│   └── cli/           # Command-line interfaces
├── tests/             # Comprehensive test suite
├── examples/          # Example models (Attention, BERT, sparse attention variants)
└── configs/           # Architecture configs
```

## Key Components

- **NoAliasDumper**: Custom YAML dumper for human-readable output (no `&id001` references)
- **EinsumRankRenamer**: BFS-based dimension label renaming for consistent einsum equations
- **EinsumGraphVisualizer**: PDF visualization of computation graphs
- **EinsumToTimeloop**: Export to Timeloop workload format
- **Node Registry**: Extensible registry for operation handlers
- **LLM Agent** (optional): Dynamic handler generation for unknown operations

## Contributing

- Follow Google's Python Style Guide
- Add tests for new features
- Update documentation for API changes
- Run `bash run_tests.sh` before submitting PRs

## Documentation

- `TESTING_GUIDE.md`: Comprehensive testing documentation

## License

MIT License
