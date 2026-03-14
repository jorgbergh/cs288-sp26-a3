#!/bin/bash
# Creates the Gradescope submission zip with only the required files.
# Usage: bash create_submission_zip.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ZIP_NAME="submission.zip"

rm -f "$ZIP_NAME"

zip -r "$ZIP_NAME" \
    run.sh \
    rag.py \
    llm.py \
    datastore/chunks.json \
    datastore/bm25_index.pkl \
    datastore/faiss_index.bin \
    datastore/embedding_model/ \
    qa_dataset.jsonl \
    iaa_results.json

echo ""
echo "Created $ZIP_NAME ($(du -h "$ZIP_NAME" | cut -f1))"
echo ""
echo "Contents:"
zipinfo -1 "$ZIP_NAME"
