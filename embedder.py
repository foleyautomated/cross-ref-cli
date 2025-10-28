"""
Text embedding and FAISS index generation for cross-ref-cli.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from config import config


class TextChunker:
    """Handles text chunking with configurable size and overlap."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

    def chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk

        Returns:
            List of tuples (chunk_text, start_position)
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append((chunk, start))

            # Move start position forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap

            # Break if we've processed all text
            if end >= text_length:
                break

        return chunks

    def chunk_file(self, file_path: str) -> List[Tuple[str, int, int]]:
        """
        Read a file and split it into overlapping chunks.

        Args:
            file_path: Path to the text file

        Returns:
            List of tuples (chunk_text, start_position, line_number)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self.chunk_text(text)

        # Calculate line numbers for each chunk
        chunks_with_lines = []
        for chunk_text, start_pos in chunks:
            # Count newlines up to this position to get line number
            line_number = text[:start_pos].count('\n') + 1
            chunks_with_lines.append((chunk_text, start_pos, line_number))

        return chunks_with_lines


class Embedder:
    """Handles embedding generation and FAISS index creation."""

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use for inference ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name or config.embedding_model
        self.device = device or config.device

        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"Model loaded on device: {self.device}")

    def embed_chunks(self, chunks: List[str], batch_size: int = None) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks: List of text chunks
            batch_size: Batch size for processing

        Returns:
            Numpy array of embeddings
        """
        batch_size = batch_size or config.batch_size
        normalize = config.normalize_embeddings

        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create a FAISS index from embeddings.

        Args:
            embeddings: Numpy array of embeddings

        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        index_type = config.index_type

        print(f"Creating FAISS index (type: {index_type}, dimension: {dimension})...")

        if index_type.lower() == 'flat':
            # Exact search using L2 distance
            index = faiss.IndexFlatL2(dimension)
        elif index_type.lower() == 'ivf':
            # Approximate search using IVF
            n_clusters = min(config.n_clusters, len(embeddings) // 10)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)

            print(f"Training IVF index with {n_clusters} clusters...")
            index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}. Use 'Flat' or 'IVF'")

        # Add embeddings to index
        index.add(embeddings)
        print(f"Added {index.ntotal} vectors to FAISS index")

        return index


def embed_document(
    file_path: str,
    model_name: str = None,
    output_path: str = None
) -> Tuple[faiss.Index, List[Tuple[str, int, int]]]:
    """
    Generate embeddings for a document and create a FAISS index.

    Args:
        file_path: Path to the text file
        model_name: Optional model name override
        output_path: Optional custom output path for FAISS index

    Returns:
        Tuple of (FAISS index, list of chunks with metadata)
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine output path
    if output_path is None:
        # Create output filename based on input file and model
        model_name_safe = (model_name or config.embedding_model).replace('/', '-')
        output_path = file_path_obj.parent / f"{file_path_obj.stem}___{model_name_safe}.faiss"

    # Chunk the text
    print(f"\nChunking text from: {file_path}")
    chunker = TextChunker()
    chunks_with_metadata = chunker.chunk_file(str(file_path_obj))

    print(f"Created {len(chunks_with_metadata)} chunks")
    print(f"  Chunk size: {config.chunk_size} characters")
    print(f"  Chunk overlap: {config.chunk_overlap} characters")

    # Extract just the text for embedding
    chunk_texts = [chunk[0] for chunk in chunks_with_metadata]

    # Generate embeddings
    embedder = Embedder(model_name=model_name)
    embeddings = embedder.embed_chunks(chunk_texts)

    # Create FAISS index
    index = embedder.create_faiss_index(embeddings)

    # Save FAISS index
    print(f"\nSaving FAISS index to: {output_path}")
    faiss.write_index(index, str(output_path))

    # Save metadata (chunks with positions and line numbers)
    metadata_path = output_path.with_suffix('.metadata.txt')
    print(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for i, (chunk_text, start_pos, line_num) in enumerate(chunks_with_metadata):
            # Escape the chunk text for storage
            chunk_escaped = chunk_text.replace('\n', '\\n').replace('\t', '\\t')
            f.write(f"{i}\t{start_pos}\t{line_num}\t{chunk_escaped}\n")

    print(f"\n✓ Successfully created embeddings!")
    print(f"  FAISS index: {output_path}")
    print(f"  Metadata: {metadata_path}")

    return index, chunks_with_metadata
