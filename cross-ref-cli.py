#!/usr/bin/env python3
"""
cross-ref-cli: A command line interface for semantically finding where one body of text quotes another.
"""

import argparse
import sys
import csv
from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np
from config import config
from embedder import embed_document as embed_doc

# Default documents directory
DOCUMENTS_DIR = Path(__file__).parent / "documents"


def resolve_file_path(file_path: str, default_dir: Path = DOCUMENTS_DIR) -> Path:
    """
    Resolve a file path, checking the documents directory if not found.

    Args:
        file_path: The file path to resolve
        default_dir: Default directory to check (default: documents/)

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If file cannot be found
    """
    path = Path(file_path)

    # If absolute path or exists in current location, return as-is
    if path.is_absolute() or path.exists():
        return path

    # Check in default documents directory
    doc_path = default_dir / file_path
    if doc_path.exists():
        return doc_path

    # Check if it's just a filename without path
    if "/" not in file_path and "\\" not in file_path:
        doc_path = default_dir / file_path
        if doc_path.exists():
            return doc_path

    # File not found anywhere
    raise FileNotFoundError(f"File not found: {file_path} (also checked in {default_dir})")


def load_faiss_index(faiss_path: str) -> Tuple[faiss.Index, List[Tuple[int, int, int, str]]]:
    """
    Load a FAISS index and its associated metadata.

    Args:
        faiss_path: Path to the .faiss file

    Returns:
        Tuple of (FAISS index, metadata list)
        Metadata format: List of tuples (chunk_id, start_pos, line_num, chunk_text)

    Raises:
        FileNotFoundError: If FAISS index or metadata file not found
    """
    faiss_path_obj = Path(faiss_path)

    if not faiss_path_obj.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    # Load FAISS index
    print(f"Loading FAISS index: {faiss_path}")
    index = faiss.read_index(str(faiss_path_obj))
    print(f"  Loaded {index.ntotal} vectors")

    # Load metadata
    metadata_path = faiss_path_obj.with_suffix('.metadata.txt')
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    print(f"Loading metadata: {metadata_path}")
    metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                chunk_id = int(parts[0])
                start_pos = int(parts[1])
                line_num = int(parts[2])
                # Unescape the chunk text
                chunk_text = parts[3].replace('\\n', '\n').replace('\\t', '\t')
                metadata.append((chunk_id, start_pos, line_num, chunk_text))

    print(f"  Loaded {len(metadata)} metadata entries")

    return index, metadata


def embed_document(
    file_path: str,
    model: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    chunk_mode: str = None
) -> None:
    """
    Generate embeddings for a target text file.

    Args:
        file_path: Path to the text file to embed
        model: The embedding model to use (default: from config)
        chunk_size: Size of each chunk (in characters or lines)
        chunk_overlap: Overlap between chunks (in characters or lines)
        chunk_mode: Chunking mode - 'character' or 'line'
    """
    # Use config default if model not specified
    if model is None:
        model = config.embedding_model

    # Default to character mode if not specified
    if chunk_mode is None:
        chunk_mode = 'character'

    print(f"=" * 60)
    print(f"Embedding: {file_path}")
    print(f"Model: {model}")
    print(f"=" * 60)

    try:
        embed_doc(
            file_path,
            model_name=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_mode=chunk_mode
        )
    except Exception as e:
        print(f"\nError during embedding: {e}", file=sys.stderr)
        raise


def compare_documents(primary_db: str, reference_db: str, output: str = None) -> None:
    """
    Compare two FAISS databases to find semantic similarities.

    Args:
        primary_db: Path to the primary FAISS database
        reference_db: Path to the reference FAISS database
        output: Optional output CSV file path
    """
    # TODO: Implement comparison logic
    print(f"Comparing {primary_db} with {reference_db}...")
    pass


def find_references(
    query_faiss: str,
    reference_faiss: str,
    output: str = None,
    top_k: int = None,
    threshold: float = None
) -> None:
    """
    Find passages in the query document that semantically reference passages in the reference document.

    Args:
        query_faiss: Path to the query document's FAISS index (e.g., Frederick Douglass)
        reference_faiss: Path to the reference document's FAISS index (e.g., KJV Bible)
        output: Output CSV file path (default: query_name_references_reference_name.csv)
        top_k: Number of top matches to return per query chunk (default: from config)
        threshold: Minimum similarity threshold 0.0-1.0 (default: from config)
    """
    # Use config defaults if not specified
    top_k = top_k or config.top_k
    threshold = threshold or config.similarity_threshold

    print(f"=" * 60)
    print(f"Finding references in query document that cite reference document")
    print(f"Query: {query_faiss}")
    print(f"Reference: {reference_faiss}")
    print(f"Top-K: {top_k}, Threshold: {threshold}")
    print(f"=" * 60)

    # Load both FAISS indices
    query_index, query_metadata = load_faiss_index(query_faiss)
    reference_index, reference_metadata = load_faiss_index(reference_faiss)

    # Prepare output path
    if output is None:
        query_name = Path(query_faiss).stem.split('___')[0]
        reference_name = Path(reference_faiss).stem.split('___')[0]
        output = Path(query_faiss).parent / f"{query_name}_references_{reference_name}.csv"
    else:
        output = Path(output)

    print(f"\nSearching for references...")
    print(f"This may take a while for large documents...")

    # Get all query embeddings from the query index
    # For IndexFlat, we can reconstruct vectors
    query_vectors = np.zeros((query_index.ntotal, query_index.d), dtype=np.float32)
    for i in range(query_index.ntotal):
        query_vectors[i] = query_index.reconstruct(i)

    # Search the reference index for each query vector
    # This finds the top_k most similar reference chunks for each query chunk
    print(f"Searching {query_index.ntotal} query chunks against {reference_index.ntotal} reference chunks...")
    distances, indices = reference_index.search(query_vectors, top_k)

    # Convert distances to similarity scores
    # For L2 distance with normalized embeddings: similarity = 1 - (distance^2 / 2)
    # This gives us a similarity score between 0 and 1
    if config.normalize_embeddings:
        similarities = 1 - (distances ** 2 / 2)
    else:
        # For non-normalized embeddings, use negative distance as similarity
        similarities = -distances

    # Collect results above threshold
    results = []
    for query_idx in range(len(query_metadata)):
        query_chunk_id, query_start_pos, query_line_num, query_text = query_metadata[query_idx]

        for k in range(top_k):
            ref_idx = indices[query_idx][k]
            similarity = similarities[query_idx][k]

            # Filter by threshold
            if similarity >= threshold:
                ref_chunk_id, ref_start_pos, ref_line_num, ref_text = reference_metadata[ref_idx]

                results.append({
                    'query_line': query_line_num,
                    'query_chunk_id': query_chunk_id,
                    'query_text': query_text[:config.max_chunk_display],
                    'reference_line': ref_line_num,
                    'reference_chunk_id': ref_chunk_id,
                    'reference_text': ref_text[:config.max_chunk_display],
                    'similarity': similarity
                })

    print(f"\nFound {len(results)} matches above threshold {threshold}")

    # Write results to CSV
    if len(results) > 0:
        print(f"Writing results to: {output}")
        with open(output, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'query_line',
                'query_chunk_id',
                'query_text',
                'reference_line',
                'reference_chunk_id',
                'reference_text',
                'similarity'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Sort by similarity (descending) then by query line number
            results.sort(key=lambda x: (-x['similarity'], x['query_line']))
            writer.writerows(results)

        print(f"\n✓ Successfully created references CSV!")
        print(f"  Output: {output}")
        print(f"  Total matches: {len(results)}")
    else:
        print(f"\nNo matches found above threshold {threshold}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Semantically find where one body of text quotes another.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate embeddings:
    cross-ref-cli.py --embed your_primary_document.txt

  Compare two documents:
    cross-ref-cli.py --compare primary.faiss reference.faiss

  Find references:
    cross-ref-cli.py --find-references douglass.faiss kjv.faiss
        """
    )

    # Main operation flags
    parser.add_argument(
        "--embed",
        metavar="FILE",
        type=str,
        help="Generate embeddings for the specified text file (searches documents/ folder by default)"
    )

    parser.add_argument(
        "--compare",
        metavar=("PRIMARY_DB", "REFERENCE_DB"),
        nargs=2,
        type=str,
        help="Compare two FAISS databases to find semantic similarities (searches documents/ folder by default)"
    )

    parser.add_argument(
        "--find-references",
        metavar=("QUERY_FAISS", "REFERENCE_FAISS"),
        nargs=2,
        type=str,
        help="Find passages in query document that reference the reference document (searches documents/ folder by default)"
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Embedding model to use (default: from .env, currently {config.embedding_model})"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for comparison results (CSV format)"
    )

    # Chunking options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Size of each chunk in characters or lines (default: from .env, currently {config.chunk_size})"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=f"Overlap between chunks in characters or lines (default: from .env, currently {config.chunk_overlap})"
    )

    parser.add_argument(
        "--chunk-mode",
        type=str,
        choices=['character', 'line'],
        default=None,
        help="Chunking mode: 'character' (default) or 'line'"
    )

    # Reference finding options
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Number of top matches per query chunk for --find-references (default: from .env, currently {config.top_k})"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Minimum similarity threshold 0.0-1.0 for --find-references (default: from .env, currently {config.similarity_threshold})"
    )

    args = parser.parse_args()

    # Validate that at least one operation is specified
    if not args.embed and not args.compare and not args.find_references:
        parser.error("Please specify either --embed, --compare, or --find-references")
        return 1

    # Execute the appropriate operation
    if args.embed:
        try:
            file_path = resolve_file_path(args.embed)
            embed_document(
                str(file_path),
                model=args.model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                chunk_mode=args.chunk_mode
            )
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if args.compare:
        primary_db, reference_db = args.compare
        try:
            primary_path = resolve_file_path(primary_db)
            reference_path = resolve_file_path(reference_db)
            compare_documents(str(primary_path), str(reference_path), output=args.output)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if args.find_references:
        query_faiss, reference_faiss = args.find_references
        try:
            query_path = resolve_file_path(query_faiss)
            reference_path = resolve_file_path(reference_faiss)
            find_references(
                str(query_path),
                str(reference_path),
                output=args.output,
                top_k=args.top_k,
                threshold=args.threshold
            )
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error during reference finding: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
