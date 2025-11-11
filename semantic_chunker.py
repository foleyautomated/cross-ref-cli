"""
Semantic text chunking that respects sentence and paragraph boundaries.
"""

import re
from typing import List, Tuple


class SemanticChunker:
    """
    Chunks text semantically by sentences and paragraphs rather than arbitrary character counts.
    Includes overlap between chunks to preserve context.
    """

    def __init__(self, max_chunk_size: int = 512, overlap_sentences: int = 2):
        """
        Initialize the semantic chunker.

        Args:
            max_chunk_size: Maximum characters per chunk (soft limit)
            overlap_sentences: Number of sentences to overlap between chunks (default: 2)
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using basic sentence boundary detection.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Basic sentence splitting - handles common cases
        # This regex splits on . ! ? followed by whitespace and capital letter
        # while avoiding common abbreviations

        # First, protect common abbreviations
        text = re.sub(r'\bMr\.', 'Mr<DOT>', text)
        text = re.sub(r'\bMrs\.', 'Mrs<DOT>', text)
        text = re.sub(r'\bMs\.', 'Ms<DOT>', text)
        text = re.sub(r'\bDr\.', 'Dr<DOT>', text)
        text = re.sub(r'\bSt\.', 'St<DOT>', text)
        text = re.sub(r'\bvs\.', 'vs<DOT>', text)
        text = re.sub(r'\betc\.', 'etc<DOT>', text)
        text = re.sub(r'\bi\.e\.', 'i<DOT>e<DOT>', text)
        text = re.sub(r'\be\.g\.', 'e<DOT>g<DOT>', text)

        # Split on sentence boundaries
        # Look for . ! ? followed by space and capital letter or quote + capital
        sentences = re.split(r'([.!?])\s+(?=[A-Z"\'\u201c\u201d])', text)

        # Reconstruct sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]

            # Restore abbreviations
            sentence = sentence.replace('<DOT>', '.')
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)

        # Handle the last sentence if it doesn't end with punctuation
        if len(sentences) > 0 and sentences[-1].strip():
            last = sentences[-1].replace('<DOT>', '.').strip()
            if last:
                result.append(last)

        return result if result else [text.strip()]

    def chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Chunk text semantically by grouping sentences with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of tuples (chunk_text, start_position)
        """
        # Split into paragraphs first (double newline)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        all_sentences = []  # Store all sentences with their positions
        current_pos = 0

        # First pass: extract all sentences with positions
        for para in paragraphs:
            if not para.strip():
                current_pos += len(para) + 2  # +2 for paragraph break
                continue

            # Split paragraph into sentences
            sentences = self.split_into_sentences(para)

            for sentence in sentences:
                if sentence.strip():
                    all_sentences.append((sentence, current_pos))
                    current_pos += len(sentence) + 1  # +1 for space

            # Account for paragraph break
            current_pos += 2

        # Second pass: create overlapping chunks
        i = 0
        while i < len(all_sentences):
            current_chunk_sentences = []
            current_chunk_length = 0
            chunk_start_pos = all_sentences[i][1]

            # Add sentences until we reach max_chunk_size
            j = i
            while j < len(all_sentences):
                sentence, _ = all_sentences[j]
                sentence_length = len(sentence) + 1

                # Check if adding this sentence would exceed limit
                if current_chunk_sentences and (current_chunk_length + sentence_length) > self.max_chunk_size:
                    break

                current_chunk_sentences.append(sentence)
                current_chunk_length += sentence_length
                j += 1

            # Create chunk
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences).strip()
                chunks.append((chunk_text, chunk_start_pos))

            # Move forward, but overlap by overlap_sentences
            # This ensures context preservation between chunks
            advance = max(1, len(current_chunk_sentences) - self.overlap_sentences)
            i += advance

        return chunks

    def chunk_file(self, file_path: str) -> List[Tuple[str, int, int]]:
        """
        Read a file and split it into semantic chunks.

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


def chunk_text_semantically(file_path: str, max_chunk_size: int = 512, overlap_sentences: int = 2) -> List[Tuple[str, int, int]]:
    """
    Chunk a text file semantically by sentences and paragraphs with overlap.

    Args:
        file_path: Path to the text file
        max_chunk_size: Maximum characters per chunk
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of tuples (chunk_text, start_position, line_number)
    """
    chunker = SemanticChunker(max_chunk_size=max_chunk_size, overlap_sentences=overlap_sentences)
    return chunker.chunk_file(file_path)
