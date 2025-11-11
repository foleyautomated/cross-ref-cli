"""
Semantic text chunking that respects sentence and paragraph boundaries.
"""

import re
from typing import List, Tuple


class SemanticChunker:
    """
    Chunks text semantically by sentences and paragraphs rather than arbitrary character counts.
    """

    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 100):
        """
        Initialize the semantic chunker.

        Args:
            max_chunk_size: Maximum characters per chunk (soft limit)
            min_chunk_size: Minimum characters per chunk (soft limit)
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

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
        Chunk text semantically by grouping sentences.

        Args:
            text: Text to chunk

        Returns:
            List of tuples (chunk_text, start_position)
        """
        # Split into paragraphs first (double newline or single newline)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_pos = 0

        for para in paragraphs:
            if not para.strip():
                current_pos += len(para)
                continue

            # Split paragraph into sentences
            sentences = self.split_into_sentences(para)

            # Group sentences into chunks
            current_chunk = []
            current_chunk_start = current_pos
            current_chunk_length = 0

            for sentence in sentences:
                sentence_length = len(sentence) + 1  # +1 for space

                # Check if adding this sentence would exceed max_chunk_size
                if current_chunk and (current_chunk_length + sentence_length) > self.max_chunk_size:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append((chunk_text, current_chunk_start))

                    # Start new chunk
                    current_chunk = [sentence]
                    current_chunk_start = current_pos
                    current_chunk_length = sentence_length
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_chunk_length += sentence_length

                current_pos += sentence_length

            # Save remaining sentences in chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append((chunk_text, current_chunk_start))

            # Account for paragraph break
            current_pos += 2  # For \n\n between paragraphs

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


def chunk_text_semantically(file_path: str, max_chunk_size: int = 512, min_chunk_size: int = 100) -> List[Tuple[str, int, int]]:
    """
    Chunk a text file semantically by sentences and paragraphs.

    Args:
        file_path: Path to the text file
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk

    Returns:
        List of tuples (chunk_text, start_position, line_number)
    """
    chunker = SemanticChunker(max_chunk_size=max_chunk_size, min_chunk_size=min_chunk_size)
    return chunker.chunk_file(file_path)
