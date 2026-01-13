#!/usr/bin/env bash
set -euo pipefail

# Run Solar processing + einsum pipeline for the Dense Attention example.
#
# Dense Attention computes the full attention matrix:
#   attention = softmax(Q @ K^T / sqrt(d_k)) @ V
#
# This is the standard attention mechanism used in Transformers.
#
# Outputs are written under:
#   solar/examples/DenseAttention/output/{graph,einsum,timeloop}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_FILE="${SCRIPT_DIR}/DenseAttention.py"
OUT_BASE="${SOLAR_DENSE_ATTN_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
GRAPH_OUT="${OUT_BASE}/graph"
EINSUM_OUT="${OUT_BASE}/einsum"
TIMELOOP_OUT="${OUT_BASE}/timeloop"

if ! mkdir -p "${GRAPH_OUT}" "${EINSUM_OUT}" "${TIMELOOP_OUT}"; then
  echo "❌ Failed to create output directories under: ${OUT_BASE}" >&2
  echo "   Tip: set SOLAR_DENSE_ATTN_OUTPUT_DIR to a writable path, e.g.:" >&2
  echo "     SOLAR_DENSE_ATTN_OUTPUT_DIR=/tmp/solar_dense_attn_output bash ${SCRIPT_DIR}/run_solar.sh" >&2
  exit 1
fi

cd "${SOLAR_ROOT}"


# Step 1: generate pytorch graph, output is pytorch_graph.yaml
echo "==> Processing model -> ${GRAPH_OUT}"
python3 -m solar.cli.process_model \
  --model-file "${MODEL_FILE}" \
  --output-dir "${GRAPH_OUT}" \
  --force-rerun

# Step 2: convert pytorch graph to einsum graph, output is einsum_graph.yaml, ignore the einsum_graph_renamed.yaml
echo "==> Converting pytorch graph -> ${EINSUM_OUT}"
python3 -m solar.cli.toeinsum_model \
  --graph-path "${GRAPH_OUT}/pytorch_graph.yaml" \
  --output-dir "${EINSUM_OUT}" \
  --no-copy-graph \
  --save-graph

# Step 3: convert einsum graph to timeloop graph, output is timeloop_graph.yaml, don't use the einsum_graph_renamed.yaml as input
echo "==> Converting to Timeloop format -> ${TIMELOOP_OUT}"
python3 -m solar.cli.totimeloop \
  --einsum-graph-path "${EINSUM_OUT}/einsum_graph.yaml" \
  --output-dir "${TIMELOOP_OUT}"

echo ""
echo "Done."
echo ""
echo "=== Dense Attention Example Outputs ==="
echo "PyTorch graph:   ${GRAPH_OUT}/pytorch_graph.yaml"
echo "Einsum graph:    ${EINSUM_OUT}/einsum_graph.yaml"
# echo "Einsum renamed:  ${EINSUM_OUT}/einsum_graph_renamed.yaml"
echo "Graph PDF:       ${EINSUM_OUT}/einsum_graph.pdf"
echo "Timeloop graph:  ${TIMELOOP_OUT}/timeloop_graph.yaml"
echo ""
echo "Dense attention mechanism:"
echo "  - Q @ K^T / sqrt(d_k)  (attention scores)"
echo "  - softmax(scores)"
echo "  - attention_weights @ V  (context)"
