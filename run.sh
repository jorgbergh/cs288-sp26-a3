#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Download large datastore files from Git LFS at runtime ---
DATASTORE_DIR="$SCRIPT_DIR/datastore"
mkdir -p "$DATASTORE_DIR"

# Your raw LFS file URLs — replace with your actual repo URLs
BASE_URL="https://media.githubusercontent.com/media/<your-username>/<your-repo>/main/datastore"

download_if_missing() {
    local filename="$1"
    local filepath="$DATASTORE_DIR/$filename"
    if [ ! -f "$filepath" ]; then
        echo "Downloading $filename..."
        curl -L "$BASE_URL/$filename" -o "$filepath"
    else
        echo "$filename already present, skipping download."
    fi
}

download_if_missing "chunks.json"
download_if_missing "bm25_index.pkl"
download_if_missing "faiss_index.bin"

# --- Run RAG ---
python3 "$SCRIPT_DIR/rag.py" "$1" "$2"