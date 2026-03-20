#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Run RAG ---
python3 "$SCRIPT_DIR/rag.py" "$1" "$2"
