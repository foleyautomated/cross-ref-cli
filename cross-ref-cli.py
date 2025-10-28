#!/usr/bin/env python3
"""
cross-ref-cli: A command line interface for semantically finding where one body of text quotes another.
"""

import argparse
import sys
from pathlib import Path
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


def embed_document(file_path: str, model: str = None) -> None:
    """
    Generate embeddings for a target text file.

    Args:
        file_path: Path to the text file to embed
        model: The embedding model to use (default: from config)
    """
    # Use config default if model not specified
    if model is None:
        model = config.embedding_model

    print(f"=" * 60)
    print(f"Embedding: {file_path}")
    print(f"Model: {model}")
    print(f"=" * 60)

    try:
        embed_doc(file_path, model_name=model)
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

    args = parser.parse_args()

    # Validate that at least one operation is specified
    if not args.embed and not args.compare:
        parser.error("Please specify either --embed or --compare")
        return 1

    # Execute the appropriate operation
    if args.embed:
        try:
            file_path = resolve_file_path(args.embed)
            embed_document(str(file_path), model=args.model)
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
