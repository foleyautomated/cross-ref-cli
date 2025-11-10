# Overview

`cross-ref-cli` is a Python-based command line interface for semantically finding where a body of text quotes or references the Bible. It uses sentence transformers to generate embeddings and FAISS for efficient vector similarity search.

## Project Structure

```
cross-ref-cli/
├── cross-ref-cli.py        # Main CLI entry point
├── embedder.py             # Text chunking and embedding generation
├── config.py               # Configuration management
├── .env                    # User configuration (git-ignored)
├── .env.example            # Configuration template
├── requirements.txt        # Python dependencies
├── venv/                   # Virtual environment (git-ignored)
├── documents/              # Default location for text files and generated databases
│   ├── .gitkeep
│   └── *.txt, *.faiss, *.metadata.txt (git-ignored)
└── README.md              # User documentation
```

## Setup

### Environment
Uses Python 3.13 with a virtual environment to manage dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # or ./venv/bin/python3 to run directly
pip install -r requirements.txt
```

### Dependencies
- `sentence-transformers` - For generating text embeddings
- `faiss-cpu` - Vector similarity search and indexing
- `numpy` - Numerical operations
- `pandas` - CSV output generation (TODO: implement comparison)
- `tqdm` - Progress bars

### Configuration
All parameters are managed via `.env` file (copy from `.env.example`):

**Embedding & Model:**
- `EMBEDDING_MODEL` - HuggingFace model (default: BAAI/bge-base-en-v1.5)
- `NORMALIZE_EMBEDDINGS` - Normalize vectors for cosine similarity (default: true)

**Text Chunking:**
- `CHUNK_SIZE` - Characters per chunk (default: 512)
- `CHUNK_OVERLAP` - Character overlap between chunks (default: 50)

**Processing:**
- `BATCH_SIZE` - Chunks per batch (default: 32)
- `MAX_TOKENS` - Max token length (default: 512)
- `DEVICE` - cpu/cuda/mps (default: cpu)

**Comparison (for future use):**
- `SIMILARITY_THRESHOLD` - Min similarity score 0.0-1.0 (default: 0.7)
- `TOP_K` - Number of top matches (default: 5)

**FAISS Index:**
- `INDEX_TYPE` - Flat (exact) or IVF (approximate) (default: Flat)
- `N_CLUSTERS` - For IVF indexing (default: 100)

**Output:**
- `INCLUDE_CHUNK_TEXT` - Include text in CSV (default: true)
- `MAX_CHUNK_DISPLAY` - Max chars per chunk (default: 200)

## Current Implementation Status

### ✅ Completed

**Embedding Generation (`--embed`):**
- Text file reading from `documents/` folder (with automatic path resolution)
- Configurable text chunking with overlap
- Sentence transformer embedding generation
- FAISS index creation (Flat and IVF supported)
- Metadata storage (chunk text, positions, line numbers)
- Progress bars and status output
- Successfully tested with kjv.txt (9,374 chunks, 27MB FAISS index)

**CLI Features:**
- Command-line argument parsing
- Path resolution (searches `documents/` folder by default)
- Configuration loading from .env
- Model override via `--model` flag
- Error handling and validation

**Infrastructure:**
- Virtual environment setup
- Dependency management
- Git configuration (.gitignore for generated files)
- Project documentation (README.md)

### 🚧 TODO

**Comparison Functionality (`--compare`):**
- Load FAISS indices and metadata
- Query primary document chunks against reference document
- Filter by similarity threshold
- Generate CSV output with:
  - Line numbers from both documents
  - Similarity scores
  - Chunk excerpts
- Save results to output file

**Additional Features:**
- Batch comparison operations
- Custom output formatting options
- Performance optimization for large documents
- Support for additional index types

## Usage

### Generate Embeddings

Place text files in the `documents/` folder:
```bash
./venv/bin/python3 cross-ref-cli.py --embed your_document.txt
```

This creates:
- `your_document___BAAI-bge-base-en-v1.5.faiss` - FAISS vector index
- `your_document___BAAI-bge-base-en-v1.5.metadata.txt` - Chunk metadata

Override model:
```bash
./venv/bin/python3 cross-ref-cli.py --embed your_document.txt --model sentence-transformers/all-MiniLM-L6-v2
```

### Compare Documents (TODO)

```bash
./venv/bin/python3 cross-ref-cli.py --compare primary.faiss reference.faiss --output results.csv
```

The CLI automatically searches the `documents/` folder for files.

## Implementation Details

### Text Chunking (embedder.py:TextChunker)
- Splits text into overlapping windows
- Maintains character position and line number metadata
- Overlap helps preserve context across chunk boundaries
- Configurable chunk size and overlap

### Embedding Generation (embedder.py:Embedder)
- Loads sentence transformer models via HuggingFace
- Processes chunks in configurable batches
- Optional embedding normalization for cosine similarity
- Supports CPU, CUDA, and MPS (Apple Silicon) devices

### FAISS Indexing
- **Flat index**: Exact nearest neighbor search (default)
- **IVF index**: Approximate search with clustering for speed
- Stores vector embeddings for efficient similarity search
- Separate metadata file maintains text chunks and positions

### Path Resolution
- Automatically searches `documents/` folder for relative paths
- Supports absolute paths and relative paths from CWD
- Clear error messages showing search locations

## Example: KJV Bible Embedding

Successful test case:
```bash
./venv/bin/python3 cross-ref-cli.py --embed kjv.txt
```

Results:
- Input: kjv.txt (text file in documents/)
- Chunks: 9,374 chunks @ 512 chars each with 50 char overlap
- Model: BAAI/bge-base-en-v1.5 (768-dimensional embeddings)
- Output: 27MB FAISS index + 4.9MB metadata
- Processing time: ~4.5 minutes on CPU

## Notes

- All generated files (.faiss, .metadata.txt) are git-ignored
- Virtual environment (venv/) is git-ignored
- User configuration (.env) is git-ignored
- Only .env.example and documents/.gitkeep are tracked
- File paths support both absolute and relative, with automatic documents/ folder resolution 