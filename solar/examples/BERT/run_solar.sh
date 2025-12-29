#!/usr/bin/env bash
set -euo pipefail

# Run Solar processing + einsum pipeline for the standalone BERT example.
#
# Outputs are written under:
#   solar/examples/BERT/output/{graph,einsum,timeloop}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_FILE="${SCRIPT_DIR}/BERT.py"
OUT_BASE="${SOLAR_BERT_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
GRAPH_OUT="${OUT_BASE}/graph"
EINSUM_OUT="${OUT_BASE}/einsum"
TIMELOOP_OUT="${OUT_BASE}/timeloop"

if ! mkdir -p "${GRAPH_OUT}" "${EINSUM_OUT}" "${TIMELOOP_OUT}"; then
  echo "❌ Failed to create output directories under: ${OUT_BASE}" >&2
  echo "   Tip: set SOLAR_BERT_OUTPUT_DIR to a writable path, e.g.:" >&2
  echo "     SOLAR_BERT_OUTPUT_DIR=/tmp/solar_bert_output bash ${SCRIPT_DIR}/run_solar.sh" >&2
  exit 1
fi

cd "${SOLAR_ROOT}"

echo "==> Processing model -> ${GRAPH_OUT}"
python3 -m solar.cli.process_model \
  --model-file "${MODEL_FILE}" \
  --output-dir "${GRAPH_OUT}" \
  --force-rerun

echo "==> Converting pytorch graph -> ${EINSUM_OUT}"
python3 -m solar.cli.toeinsum_model \
  --graph-path "${GRAPH_OUT}/pytorch_graph.yaml" \
  --output-dir "${EINSUM_OUT}" \
  --no-copy-graph \
  --save-graph

echo "==> Converting to Timeloop format -> ${TIMELOOP_OUT}"
python3 -m solar.cli.totimeloop \
  --einsum-graph-path "${EINSUM_OUT}/einsum_graph_renamed.yaml" \
  --output-dir "${TIMELOOP_OUT}"

echo ""
echo "Done."
echo ""
echo "=== BERT Example Outputs ==="
echo "PyTorch graph:   ${GRAPH_OUT}/pytorch_graph.yaml"
echo "Einsum graph:    ${EINSUM_OUT}/einsum_graph.yaml"
echo "Einsum renamed:  ${EINSUM_OUT}/einsum_graph_renamed.yaml"
echo "Graph PDF:       ${EINSUM_OUT}/einsum_graph.pdf"
echo "Timeloop graph:  ${TIMELOOP_OUT}/timeloop_graph.yaml"
