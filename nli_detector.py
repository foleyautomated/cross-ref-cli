"""
Natural Language Inference (NLI) based biblical allusion detection.

Uses roberta-large-mnli to detect entailment relationships between text
and biblical verses, identifying potential allusions and references.

Supports hybrid mode: semantic similarity prefiltering + NLI validation.
"""

from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import faiss


class NLIDetector:
    """Detects biblical allusions using Natural Language Inference."""

    def __init__(self, model_name: str = "roberta-large-mnli", device: str = None):
        """
        Initialize the NLI detector.

        Args:
            model_name: HuggingFace model identifier (default: roberta-large-mnli)
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

        print(f"Loading NLI model: {self.model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully")

    def check_entailment(
        self,
        premise: str,
        hypothesis: str,
        return_all_scores: bool = False
    ) -> Dict[str, float]:
        """
        Check if premise entails hypothesis using NLI.

        Args:
            premise: The text to analyze (e.g., Douglass paragraph)
            hypothesis: The reference text (e.g., Bible verse)
            return_all_scores: If True, return all three scores (contradiction, neutral, entailment)

        Returns:
            Dictionary with scores: {'entailment': float, 'neutral': float, 'contradiction': float}
            or just {'entailment': float} if return_all_scores=False
        """
        # Tokenize input pair
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # roberta-large-mnli outputs: [contradiction, neutral, entailment]
        scores = probs[0].cpu().numpy()

        result = {
            'contradiction': float(scores[0]),
            'neutral': float(scores[1]),
            'entailment': float(scores[2])
        }

        if return_all_scores:
            return result
        else:
            return {'entailment': result['entailment']}

    def find_allusions_hybrid(
        self,
        query_paragraphs: List[Tuple[str, int, int]],  # (text, start_pos, line_num)
        bible_verses: List[Tuple[str, int, int, str]],  # (text, start_pos, line_num, reference)
        query_faiss_index: faiss.Index,
        bible_faiss_index: faiss.Index,
        semantic_candidates: int = 100,
        entailment_threshold: float = 0.5,
        early_stop_count: int = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find biblical allusions using hybrid approach: semantic search + NLI validation.

        Stage 1: Use FAISS to find top N semantically similar candidates
        Stage 2: Apply NLI to validate entailment on those candidates only

        Args:
            query_paragraphs: List of paragraphs to analyze
            bible_verses: List of Bible verses
            query_faiss_index: FAISS index for query paragraphs
            bible_faiss_index: FAISS index for Bible verses
            semantic_candidates: Number of semantic similarity candidates per paragraph (default: 100)
            entailment_threshold: Minimum entailment score (default: 0.5)
            early_stop_count: Stop after finding this many high-confidence matches per paragraph (default: None = top_k)
            top_k: Number of top matches to keep per paragraph (default: 5)

        Returns:
            List of dictionaries with allusion information
        """
        if early_stop_count is None:
            early_stop_count = top_k

        print(f"\n🔍 HYBRID MODE: Semantic Search + NLI Validation")
        print(f"Query paragraphs: {len(query_paragraphs)}")
        print(f"Bible verses: {len(bible_verses)}")
        print(f"Stage 1: Finding top {semantic_candidates} semantic candidates per paragraph")
        print(f"Stage 2: NLI validation with entailment threshold {entailment_threshold}")
        print(f"Early stopping: After {early_stop_count} high-confidence matches")
        print(f"Total NLI checks: ~{len(query_paragraphs) * semantic_candidates} (vs {len(query_paragraphs) * len(bible_verses)} without hybrid)")

        results = []

        # Stage 1: Get all query embeddings
        print("\n[Stage 1] Extracting query embeddings from FAISS...")
        query_vectors = np.zeros((query_faiss_index.ntotal, query_faiss_index.d), dtype=np.float32)
        for i in range(query_faiss_index.ntotal):
            query_vectors[i] = query_faiss_index.reconstruct(i)

        # Stage 1: Semantic search to find top candidates
        print(f"[Stage 1] Performing semantic search for {len(query_paragraphs)} paragraphs...")
        distances, indices = bible_faiss_index.search(query_vectors, semantic_candidates)

        # Stage 2: NLI validation on candidates
        print(f"\n[Stage 2] Applying NLI to validate {len(query_paragraphs) * semantic_candidates} candidates...")
        for para_idx, (para_text, para_pos, para_line) in enumerate(tqdm(query_paragraphs, desc="NLI validation")):
            paragraph_matches = []
            high_confidence_count = 0

            # Get top semantic candidates for this paragraph
            candidate_indices = indices[para_idx]

            # Apply NLI to each candidate
            for bible_idx in candidate_indices:
                verse_text, verse_pos, verse_line, verse_ref = bible_verses[bible_idx]

                # NLI check
                scores = self.check_entailment(para_text, verse_text, return_all_scores=True)
                entailment_score = scores['entailment']

                # Keep matches above threshold
                if entailment_score >= entailment_threshold:
                    paragraph_matches.append({
                        'query_text': para_text,
                        'query_line': para_line,
                        'query_pos': para_pos,
                        'verse_text': verse_text,
                        'verse_reference': verse_ref,
                        'verse_line': verse_line,
                        'entailment_score': entailment_score,
                        'neutral_score': scores['neutral'],
                        'contradiction_score': scores['contradiction']
                    })

                    high_confidence_count += 1

                    # Early stopping: if we found enough high-confidence matches, stop
                    if high_confidence_count >= early_stop_count:
                        break

            # Sort by entailment score and keep top-K
            paragraph_matches.sort(key=lambda x: x['entailment_score'], reverse=True)
            results.extend(paragraph_matches[:top_k])

        # Sort all results by entailment score
        results.sort(key=lambda x: x['entailment_score'], reverse=True)

        print(f"\n✓ Found {len(results)} allusions above threshold {entailment_threshold}")

        return results

    def find_allusions(
        self,
        query_paragraphs: List[Tuple[str, int, int]],  # (text, start_pos, line_num)
        bible_verses: List[Tuple[str, int, int, str]],  # (text, start_pos, line_num, reference)
        entailment_threshold: float = 0.5,
        top_k: int = 5,
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Find biblical allusions in query text using NLI (brute force - slow).

        NOTE: This checks every paragraph against every verse. For large datasets,
        use find_allusions_hybrid() instead.

        Args:
            query_paragraphs: List of paragraphs to analyze
            bible_verses: List of Bible verses to check against
            entailment_threshold: Minimum entailment score (default: 0.5)
            top_k: Number of top matches to keep per paragraph (default: 5)
            batch_size: Not used currently (for future batching optimization)

        Returns:
            List of dictionaries with allusion information
        """
        print(f"\n⚠️  WARNING: Using brute-force NLI (slow for large datasets)")
        print(f"Query paragraphs: {len(query_paragraphs)}")
        print(f"Bible verses: {len(bible_verses)}")
        print(f"Total NLI checks: {len(query_paragraphs) * len(bible_verses)}")
        print(f"Entailment threshold: {entailment_threshold}")
        print(f"Top-K per paragraph: {top_k}")

        results = []

        # For each query paragraph, check against all Bible verses
        for para_idx, (para_text, para_pos, para_line) in enumerate(tqdm(query_paragraphs, desc="Analyzing paragraphs")):
            paragraph_matches = []

            # Check this paragraph against each Bible verse
            for verse_text, verse_pos, verse_line, verse_ref in bible_verses:
                # Use paragraph as premise, verse as hypothesis
                # This checks if the paragraph content relates to/alludes to the verse
                scores = self.check_entailment(para_text, verse_text, return_all_scores=True)

                entailment_score = scores['entailment']

                # Keep matches above threshold
                if entailment_score >= entailment_threshold:
                    paragraph_matches.append({
                        'query_text': para_text,
                        'query_line': para_line,
                        'query_pos': para_pos,
                        'verse_text': verse_text,
                        'verse_reference': verse_ref,
                        'verse_line': verse_line,
                        'entailment_score': entailment_score,
                        'neutral_score': scores['neutral'],
                        'contradiction_score': scores['contradiction']
                    })

            # Sort by entailment score and keep top-K
            paragraph_matches.sort(key=lambda x: x['entailment_score'], reverse=True)
            results.extend(paragraph_matches[:top_k])

        # Sort all results by entailment score
        results.sort(key=lambda x: x['entailment_score'], reverse=True)

        print(f"\nFound {len(results)} allusions above threshold {entailment_threshold}")

        return results


def detect_biblical_allusions(
    query_text_file: str,
    bible_file: str,
    output_csv: str = None,
    entailment_threshold: float = 0.5,
    top_k: int = 5,
    semantic_candidates: int = 100,
    early_stop_count: int = None,
    use_hybrid: bool = True
) -> List[Dict]:
    """
    Detect biblical allusions in a text file using NLI.

    Supports hybrid mode: uses FAISS indices for semantic search prefiltering,
    then applies NLI validation only to top candidates.

    Args:
        query_text_file: Path to the text file to analyze
        bible_file: Path to the Bible text file
        output_csv: Optional output CSV file path
        entailment_threshold: Minimum entailment score
        top_k: Number of top matches per paragraph
        semantic_candidates: Number of semantic candidates to check with NLI (hybrid mode only)
        early_stop_count: Stop after finding this many matches per paragraph
        use_hybrid: Use hybrid mode if FAISS indices available (default: True)

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
    query_faiss_path = None
    bible_faiss_path = None
    query_faiss_index = None
    bible_faiss_index = None

    if use_hybrid:
        # Look for FAISS indices
        query_file = Path(query_text_file)
        bible_file_path = Path(bible_file)

        # Expected FAISS index names (based on embed_document naming convention)
        query_faiss_path = query_file.parent / f"{query_file.stem}___BAAI-bge-base-en-v1.5.faiss"
        bible_faiss_path = bible_file_path.parent / f"{bible_file_path.stem}___BAAI-bge-base-en-v1.5.faiss"

        if query_faiss_path.exists() and bible_faiss_path.exists():
            print(f"\n✓ Found FAISS indices - using HYBRID MODE")
            print(f"  Query index: {query_faiss_path}")
            print(f"  Bible index: {bible_faiss_path}")

            query_faiss_index = faiss.read_index(str(query_faiss_path))
            bible_faiss_index = faiss.read_index(str(bible_faiss_path))
        else:
            print(f"\n⚠️  FAISS indices not found - falling back to brute-force mode")
            if not query_faiss_path.exists():
                print(f"  Missing: {query_faiss_path}")
                print(f"  Run: cross-ref-cli.py --embed {query_file.name} --chunk-mode semantic")
            if not bible_faiss_path.exists():
                print(f"  Missing: {bible_faiss_path}")
                print(f"  Run: cross-ref-cli.py --embed {bible_file_path.name} --chunk-mode verse")
            use_hybrid = False

    # Initialize NLI detector
    detector = NLIDetector()

    # Find allusions (hybrid or brute-force)
    if use_hybrid and query_faiss_index and bible_faiss_index:
        results = detector.find_allusions_hybrid(
            query_paragraphs=query_chunks,
            bible_verses=bible_chunks,
            query_faiss_index=query_faiss_index,
            bible_faiss_index=bible_faiss_index,
            semantic_candidates=semantic_candidates,
            entailment_threshold=entailment_threshold,
            early_stop_count=early_stop_count,
            top_k=top_k
        )
    else:
        results = detector.find_allusions(
            query_paragraphs=query_chunks,
            bible_verses=bible_chunks,
            entailment_threshold=entailment_threshold,
            top_k=top_k
        )

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
                'entailment_score',
                'neutral_score',
                'contradiction_score'
            ])
            writer.writeheader()

            for result in results:
                writer.writerow({
                    'query_line': result['query_line'],
                    'query_text_preview': result['query_text'][:200],  # First 200 chars
                    'verse_reference': result['verse_reference'],
                    'verse_text_preview': result['verse_text'][:200],
                    'entailment_score': f"{result['entailment_score']:.4f}",
                    'neutral_score': f"{result['neutral_score']:.4f}",
                    'contradiction_score': f"{result['contradiction_score']:.4f}"
                })

        print(f"✓ Saved {len(results)} allusions to {output_path}")

    return results
