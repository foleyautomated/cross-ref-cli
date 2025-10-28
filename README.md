# cross-ref-cli

A Python-based command line interface for semantically finding where one body of text quotes another.

## Installation

1. Clone this repository
2. Copy `.env.example` to `.env` and configure as needed:
   ```bash
   cp .env.example .env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

```
cross-ref-cli/
├── documents/          # Default location for input text files and FAISS databases
├── cross-ref-cli.py    # Main CLI script
├── config.py           # Configuration management
├── .env                # Your configuration (not tracked by git)
└── .env.example        # Configuration template
```

The `documents/` folder is the default location for all input files. When you specify a filename without a full path, the CLI will automatically look in this folder.

## Configuration

All configuration parameters are managed through the `.env` file. You can customize:

### Embedding Model
- `EMBEDDING_MODEL`: HuggingFace model identifier (default: `BAAI/bge-base-en-v1.5`)

### Text Chunking
- `CHUNK_SIZE`: Size of each text chunk in characters (default: `512`)
- `CHUNK_OVERLAP`: Character overlap between chunks (default: `50`)

### Processing
- `BATCH_SIZE`: Chunks per batch (default: `32`)
- `MAX_TOKENS`: Maximum token length (default: `512`)
- `DEVICE`: Computing device - `cpu`, `cuda`, or `mps` (default: `cpu`)
- `NORMALIZE_EMBEDDINGS`: Normalize embeddings (default: `true`)

### Comparison
- `SIMILARITY_THRESHOLD`: Minimum similarity score 0.0-1.0 (default: `0.7`)
- `TOP_K`: Top matches per query (default: `5`)

### FAISS Index
- `INDEX_TYPE`: `Flat` for exact search or `IVF` for approximate (default: `Flat`)
- `N_CLUSTERS`: Clusters for IVF index (default: `100`)

### Output
- `INCLUDE_CHUNK_TEXT`: Include chunk text in CSV output (default: `true`)
- `MAX_CHUNK_DISPLAY`: Max characters per chunk in output (default: `200`)

## Usage

### Generate Embeddings

Place your text files in the `documents/` folder, then run:

```bash
python3 cross-ref-cli.py --embed your_document.txt
```

This creates a FAISS database file in `documents/` named: `your_document___MODEL-NAME.faiss`

You can also use absolute or relative paths:
```bash
python3 cross-ref-cli.py --embed /path/to/your_document.txt
```

Override the model from command line:
```bash
python3 cross-ref-cli.py --embed your_document.txt --model sentence-transformers/all-MiniLM-L6-v2
```

### Compare Documents

```bash
python3 cross-ref-cli.py --compare primary.faiss reference.faiss
```

The CLI will automatically look for `.faiss` files in the `documents/` folder.

Optional: Specify output file:
```bash
python3 cross-ref-cli.py --compare primary.faiss reference.faiss --output results.csv
```

## Output Format

The comparison produces a CSV file with:
- Line numbers from both documents
- Similarity scores
- Chunk excerpts (if enabled in config)

## Examples

1. Place your text files in `documents/` folder:
   ```bash
   cp ~/my_files/document1.txt documents/
   cp ~/my_files/document2.txt documents/
   ```

2. Embed both documents:
   ```bash
   python3 cross-ref-cli.py --embed document1.txt
   python3 cross-ref-cli.py --embed document2.txt
   ```

3. Find cross-references:
   ```bash
   python3 cross-ref-cli.py --compare document1___BAAI-bge-base-en-v1.5.faiss document2___BAAI-bge-base-en-v1.5.faiss --output matches.csv
   ```

The generated FAISS files and output CSV will be saved in the `documents/` folder alongside your text files.

## License

See LICENSE file for details.
