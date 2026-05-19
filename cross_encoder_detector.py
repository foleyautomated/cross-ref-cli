"""
Cross-encoder based biblical allusion detection.

Uses cross-encoder models (e.g., ms-marco) to score semantic similarity
between text passages and biblical verses. More accurate than NLI for
finding related passages.
"""

from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder
import torch
from tqdm import tqdm
import faiss


class CrossEncoderDetector:
    """Detects biblical allusions using cross-encoder models."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        """
        Initialize the cross-encoder detector.

        Args:
            model_name: HuggingFace cross-encoder model (default: ms-marco-MiniLM-L-6-v2)
            device: Device to use for inference ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Loading cross-encoder model: {self.model_name}")
        print(f"Device: {self.device}")

        # Load cross-encoder model
        self.model = CrossEncoder(self.model_name, device=self.device)

        print(f"Model loaded successfully")

    def score_pairs(self, text_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Score semantic similarity for text pairs using cross-encoder.

        Args:
            text_pairs: List of (text1, text2) tuples to score

        Returns:
            Numpy array of similarity scores
        """
        # Cross-encoder processes pairs together for better accuracy
        scores = self.model.predict(text_pairs, show_progress_bar=False)
        return scores

    def find_allusions_hybrid(
        self,
        query_paragraphs: List[Tuple[str, int, int]],  # (text, start_pos, line_num)
        bible_verses: List[Tuple[str, int, int, str]],  # (text, start_pos, line_num, reference)
        query_faiss_index: faiss.Index,
        bible_faiss_index: faiss.Index,
        semantic_candidates: int = 100,
        similarity_threshold: float = 0.5,
        early_stop_count: int = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find biblical allusions using hybrid approach: semantic search + cross-encoder reranking.

        Stage 1: Use FAISS to find top N semantically similar candidates
        Stage 2: Apply cross-encoder to rerank those candidates

        Args:
            query_paragraphs: List of paragraphs to analyze
            bible_verses: List of Bible verses
            query_faiss_index: FAISS index for query paragraphs
            bible_faiss_index: FAISS index for Bible verses
            semantic_candidates: Number of semantic similarity candidates per paragraph (default: 100)
            similarity_threshold: Minimum similarity score (default: 0.5)
            early_stop_count: Stop after finding this many high-confidence matches per paragraph
            top_k: Number of top matches to keep per paragraph (default: 5)

        Returns:
            List of dictionaries with allusion information
        """
        if early_stop_count is None:
            early_stop_count = top_k

        print(f"\n🔍 HYBRID MODE: Semantic Search + Cross-Encoder Reranking")
        print(f"Query paragraphs: {len(query_paragraphs)}")
        print(f"Bible verses: {len(bible_verses)}")
        print(f"Stage 1: Finding top {semantic_candidates} semantic candidates per paragraph")
        print(f"Stage 2: Cross-encoder reranking with similarity threshold {similarity_threshold}")
        print(f"Early stopping: After {early_stop_count} high-confidence matches")
        print(f"Total cross-encoder scores: ~{len(query_paragraphs) * semantic_candidates} (vs {len(query_paragraphs) * len(bible_verses)} without hybrid)")

        results = []

        # Stage 1: Get all query embeddings
        print("\n[Stage 1] Extracting query embeddings from FAISS...")
        query_vectors = np.zeros((query_faiss_index.ntotal, query_faiss_index.d), dtype=np.float32)
        for i in range(query_faiss_index.ntotal):
            query_vectors[i] = query_faiss_index.reconstruct(i)

        # Stage 1: Semantic search to find top candidates
        print(f"[Stage 1] Performing semantic search for {len(query_paragraphs)} paragraphs...")
        distances, indices = bible_faiss_index.search(query_vectors, semantic_candidates)

        # Stage 2: Cross-encoder reranking on candidates
        print(f"\n[Stage 2] Applying cross-encoder to rerank {len(query_paragraphs) * semantic_candidates} candidates...")
        for para_idx, (para_text, para_pos, para_line) in enumerate(tqdm(query_paragraphs, desc="Cross-encoder reranking")):
            # Get top semantic candidates for this paragraph
            candidate_indices = indices[para_idx]

            # Prepare text pairs for cross-encoder
            text_pairs = []
            candidate_verses = []

            for bible_idx in candidate_indices:
                verse_text, verse_pos, verse_line, verse_ref = bible_verses[bible_idx]
                text_pairs.append((para_text, verse_text))
                candidate_verses.append((verse_text, verse_pos, verse_line, verse_ref))

            # Score all pairs with cross-encoder
            scores = self.score_pairs(text_pairs)

            # Collect matches above threshold
            paragraph_matches = []
            for i, (score, (verse_text, verse_pos, verse_line, verse_ref)) in enumerate(zip(scores, candidate_verses)):
                if score >= similarity_threshold:
                    paragraph_matches.append({
                        'query_text': para_text,
                        'query_line': para_line,
                        'query_pos': para_pos,
                        'verse_text': verse_text,
                        'verse_reference': verse_ref,
                        'verse_line': verse_line,
                        'similarity_score': float(score)
                    })

                # Early stopping check
                if len(paragraph_matches) >= early_stop_count:
                    break

            # Sort by similarity score and keep top-K
            paragraph_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            results.extend(paragraph_matches[:top_k])

        # Sort all results by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        print(f"\n✓ Found {len(results)} allusions above threshold {similarity_threshold}")

        return results


def detect_biblical_allusions_crossencoder(
    query_text_file: str,
    bible_file: str,
    output_csv: str = None,
    similarity_threshold: float = 0.5,
    top_k: int = 5,
    semantic_candidates: int = 100,
    early_stop_count: int = None,
    use_hybrid: bool = True,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> List[Dict]:
    """
    Detect biblical allusions in a text file using cross-encoder.

    Supports hybrid mode: uses FAISS indices for semantic search prefiltering,
    then applies cross-encoder reranking only to top candidates.

    Args:
        query_text_file: Path to the text file to analyze
        bible_file: Path to the Bible text file
        output_csv: Optional output CSV file path
        similarity_threshold: Minimum similarity score
        top_k: Number of top matches per paragraph
        semantic_candidates: Number of semantic candidates to rerank (hybrid mode only)
        early_stop_count: Stop after finding this many matches per paragraph
        use_hybrid: Use hybrid mode if FAISS indices available (default: True)
        model_name: Cross-encoder model to use

    Returns:
        List of allusion dictionaries
    """
    from semantic_chunker import chunk_text_semantically
    from bible_parser import chunk_bible_by_verse
    import csv

    # Load query text (Douglass)
    print(f"\nLoading query text: {query_text_file}")
    query_chunks = chunk_text_semantically(
        query_text_file,
        paragraph_mode=True  # Use paragraph mode
    )
    print(f"Loaded {len(query_chunks)} paragraphs")

    # Load Bible verses
    print(f"\nLoading Bible verses: {bible_file}")
    bible_chunks = chunk_bible_by_verse(bible_file)
    print(f"Loaded {len(bible_chunks)} verses")

    # Try to load FAISS indices for hybrid mode
    query_faiss_index = None
    bible_faiss_index = None

    if use_hybrid:
        # Look for FAISS indices
        query_file = Path(query_text_file)
        bible_file_path = Path(bible_file)

        # Expected FAISS index names
        query_faiss_path = query_file.parent / f"{query_file.stem}___BAAI-bge-base-en-v1.5.faiss"
        bible_faiss_path = bible_file_path.parent / f"{bible_file_path.stem}___BAAI-bge-base-en-v1.5.faiss"

        if query_faiss_path.exists() and bible_faiss_path.exists():
            print(f"\n✓ Found FAISS indices - using HYBRID MODE")
            print(f"  Query index: {query_faiss_path}")
            print(f"  Bible index: {bible_faiss_path}")

            query_faiss_index = faiss.read_index(str(query_faiss_path))
            bible_faiss_index = faiss.read_index(str(bible_faiss_path))
        else:
            print(f"\n⚠️  FAISS indices not found - cross-encoder requires hybrid mode")
            print(f"  Missing indices - please generate them first with --embed")
            return []

    # Initialize cross-encoder detector
    detector = CrossEncoderDetector(model_name=model_name)

    # Find allusions using hybrid mode
    if query_faiss_index and bible_faiss_index:
        results = detector.find_allusions_hybrid(
            query_paragraphs=query_chunks,
            bible_verses=bible_chunks,
            query_faiss_index=query_faiss_index,
            bible_faiss_index=bible_faiss_index,
            semantic_candidates=semantic_candidates,
            similarity_threshold=similarity_threshold,
            early_stop_count=early_stop_count,
            top_k=top_k
        )
    else:
        print("❌ Hybrid mode required for cross-encoder")
        return []

    # Save to CSV if requested
    if output_csv:
        output_path = Path(output_csv)
        print(f"\nSaving results to: {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'query_line',
                'query_text_preview',
                'verse_reference',
                'verse_text_preview',
                'similarity_score'
            ])
            writer.writeheader()

            for result in results:
                writer.writerow({
                    'query_line': result['query_line'],
                    'query_text_preview': result['query_text'][:200],  # First 200 chars
                    'verse_reference': result['verse_reference'],
                    'verse_text_preview': result['verse_text'][:200],
                    'similarity_score': f"{result['similarity_score']:.4f}"
                })

        print(f"✓ Saved {len(results)} allusions to {output_path}")

    return results
