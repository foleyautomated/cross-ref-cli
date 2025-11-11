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
Uses Python 3.11 with a virtual environment to manage dependencies:

**IMPORTANT:** Python 3.11 is required for stability. Python 3.13 has multiprocessing issues with sentence-transformers.

```bash
# Install Python 3.11
brew install python@3.11

# Create virtual environment
/opt/homebrew/bin/python3.11 -m venv venv

# Install dependencies
./venv/bin/pip install -r requirements.txt
```

The embedder uses single-threaded execution to avoid segmentation faults on macOS Apple Silicon.

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
- **Verse-based chunking** (`--chunk-mode verse`) - Parses Bible text into individual verses with proper references
- **Semantic chunking** (`--chunk-mode semantic`) - Sentence-based chunking that respects semantic boundaries
- Character-based and line-based chunking (original modes)
- Sentence transformer embedding generation
- FAISS index creation (Flat and IVF supported)
- **Enhanced metadata storage** - Includes verse references (e.g., "Genesis 1:1") for Bible texts
- Status output (progress bars disabled for stability)
- Successfully tested with:
  - kjv.txt: 30,682 verses with verse references (90MB FAISS index)
  - Frederick Douglass: 2,358 semantic chunks (sentence-based)

**Reference Finding (`--find-references`):**
- Load FAISS indices and metadata (backward compatible with old format)
- Query chunks against reference document
- Filter by similarity threshold
- **CSV output with verse references** - Shows "Genesis 1:1" instead of just line numbers
- Automatic output file naming
- Successfully tested: Found 4,180 Bible references in Douglass text

**CLI Features:**
- Command-line argument parsing
- Path resolution (searches `documents/` folder by default)
- Configuration loading from .env
- Model override via `--model` flag
- **Chunk mode selection** (`--chunk-mode verse|semantic|character|line`)
- Threshold and top-k configuration for reference finding
- Error handling and validation

**Infrastructure:**
- Python 3.11 environment (downgraded from 3.13 for stability)
- Single-threaded execution to avoid multiprocessing issues
- Dependency management
- Git configuration (.gitignore for generated files)
- Project documentation (README.md, CLAUDE.md)

### 🚧 TODO

**Bible Verse Parsing Improvements:**
- Fix book name parsing for multi-word books (e.g., "Isaiah", "1 Corinthians")
- Currently some book names are truncated in verse references

**Additional Features:**
- Batch comparison operations
- Custom output formatting options
- Performance optimization for large documents (currently single-threaded)
- Support for additional FAISS index types
- Resume interrupted embedding generation
- Export results to other formats (JSON, HTML)

## Usage

### Generate Embeddings

**Bible text with verse-based chunking:**
```bash
./venv/bin/python3 cross-ref-cli.py --embed kjv.txt --chunk-mode verse
```

**Regular text with semantic chunking:**
```bash
./venv/bin/python3 cross-ref-cli.py --embed your_document.txt --chunk-mode semantic
```

**Character-based chunking (default):**
```bash
./venv/bin/python3 cross-ref-cli.py --embed your_document.txt
```

This creates:
- `your_document___BAAI-bge-base-en-v1.5.faiss` - FAISS vector index
- `your_document___BAAI-bge-base-en-v1.5.metadata.txt` - Chunk metadata (with verse references for Bible texts)

Override model:
```bash
./venv/bin/python3 cross-ref-cli.py --embed your_document.txt --model sentence-transformers/all-MiniLM-L6-v2
```

### Find Bible References

Find where a text quotes or references the Bible:
```bash
./venv/bin/python3 cross-ref-cli.py --find-references douglass.faiss kjv.faiss --threshold 0.75 --output results.csv
```

The output CSV includes:
- `query_line` - Line number in query document
- `query_text` - Text from query document
- `reference_verse` - Full Bible verse reference (e.g., "Genesis 1:1")
- `reference_text` - Text of the Bible verse
- `similarity` - Similarity score (0.0-1.0)

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

## Examples

### KJV Bible Embedding (Verse Mode)

```bash
./venv/bin/python3 cross-ref-cli.py --embed kjv.txt --chunk-mode verse
```

Results:
- Input: kjv.txt (text file in documents/)
- Chunks: 30,682 verses with proper references (e.g., "Genesis 1:1")
- Model: BAAI/bge-base-en-v1.5 (768-dimensional embeddings)
- Output: 90MB FAISS index + 4.7MB metadata
- Processing time: ~8 minutes on CPU (single-threaded)

### Frederick Douglass Text (Semantic Mode)

```bash
./venv/bin/python3 cross-ref-cli.py --embed Frederick_Douglass_Chapters.txt --chunk-mode semantic
```

Results:
- Input: Frederick Douglass autobiography chapters
- Chunks: 2,358 semantic chunks (sentence-based, respecting paragraph boundaries)
- Model: BAAI/bge-base-en-v1.5
- Output: 7.2MB FAISS index
- Processing time: ~2 minutes on CPU

### Finding Bible References

```bash
./venv/bin/python3 cross-ref-cli.py --find-references Frederick_Douglass_Chapters___BAAI-bge-base-en-v1.5.faiss kjv___BAAI-bge-base-en-v1.5.faiss --threshold 0.75
```

Results:
- Found 4,180 Bible references in Douglass text
- Output includes verse references like "John 9:16", "Daniel 8:17", "Genesis 19:19"
- Highest similarity score: 0.917 (reference to Sabbath/Pharisees)

## Notes

- All generated files (.faiss, .metadata.txt) are git-ignored
- Virtual environment (venv/) is git-ignored
- User configuration (.env) is git-ignored
- Only .env.example and documents/.gitkeep are tracked
- File paths support both absolute and relative, with automatic documents/ folder resolution 