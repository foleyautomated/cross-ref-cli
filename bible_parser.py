"""
Bible verse parser for extracting individual verses with proper references.
"""

import re
from typing import List, Tuple, Dict


class BibleParser:
    """Parses Bible text files and extracts verses with proper book/chapter/verse references."""

    # Common book title patterns in KJV format
    BOOK_PATTERNS = {
        'Genesis': r'The First Book of Moses.*Genesis',
        'Exodus': r'The Second Book of Moses.*Exodus',
        'Leviticus': r'The Third Book of Moses.*Leviticus',
        'Numbers': r'The Fourth Book of Moses.*Numbers',
        'Deuteronomy': r'The Fifth Book of Moses.*Deuteronomy',
        'Matthew': r'The Gospel According to.*Matthew',
        'Mark': r'The Gospel According to.*Mark',
        'Luke': r'The Gospel According to.*Luke',
        'John': r'(?:^The Gospel According to.*John|^The Gospel of .*John)',
        'Acts': r'The Acts of the Apostles',
        'Revelation': r'The Revelation of.*John',
    }

    def __init__(self):
        """Initialize the Bible parser."""
        self.current_book = None

    def extract_book_name(self, text: str) -> str:
        """
        Extract book name from a title line.

        Args:
            text: Line containing book title

        Returns:
            Clean book name or None
        """
        text_stripped = text.strip()

        # Check known patterns
        for book_name, pattern in self.BOOK_PATTERNS.items():
            if re.search(pattern, text_stripped, re.IGNORECASE):
                return book_name

        # Check for "The Book of X" pattern
        match = re.search(r'The Book of (\w+)', text_stripped)
        if match:
            return match.group(1)

        # Check for "The X Epistle" pattern
        match = re.search(r'The (?:First|Second|Third|1|2|3)?\s*Epistle.*of.*(\w+)', text_stripped)
        if match:
            return match.group(1)

        return None

    def parse_bible_file(self, file_path: str) -> List[Dict]:
        """
        Parse a Bible text file and extract verses with references.

        Args:
            file_path: Path to the Bible text file

        Returns:
            List of dictionaries with verse information:
            {
                'book': str,
                'chapter': int,
                'verse': int,
                'text': str,
                'reference': str (e.g., "Genesis 1:1"),
                'start_pos': int,
                'line_num': int
            }
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        verses = []
        current_book = None
        char_pos = 0

        lines = content.split('\n')

        for line_num, line in enumerate(lines, start=1):
            line_stripped = line.strip()

            # Skip empty lines
            if not line_stripped:
                char_pos += len(line) + 1
                continue

            # Check if this is a book title line
            book_name = self.extract_book_name(line_stripped)
            if book_name:
                current_book = book_name
                char_pos += len(line) + 1
                continue

            # Find all verse references in this line (both at start and inline)
            # Pattern: chapter:verse followed by space
            verse_pattern = r'(\d+):(\d+)\s+'
            matches = list(re.finditer(verse_pattern, line))

            if matches and current_book:
                # Process each verse found in this line
                for i, match in enumerate(matches):
                    chapter = int(match.group(1))
                    verse = int(match.group(2))

                    # Get the verse text starting after the reference
                    verse_start = match.end()

                    # Verse text continues until the next verse reference or end of line
                    if i + 1 < len(matches):
                        # There's another verse on this line
                        verse_end = matches[i + 1].start()
                        verse_text = line[verse_start:verse_end].strip()
                        # Remove trailing colons or punctuation before next verse
                        verse_text = re.sub(r'[:\s]+$', '', verse_text)
                    else:
                        # This is the last verse on the line, take rest of line
                        verse_text = line[verse_start:].strip()

                    # Calculate character position of this verse
                    verse_char_pos = char_pos + match.start()

                    verses.append({
                        'book': current_book,
                        'chapter': chapter,
                        'verse': verse,
                        'text': verse_text,
                        'reference': f"{current_book} {chapter}:{verse}",
                        'start_pos': verse_char_pos,
                        'line_num': line_num
                    })

            char_pos += len(line) + 1  # +1 for newline

        # Merge verse continuations (verses that span multiple lines)
        merged_verses = self._merge_verse_continuations(verses, lines)

        return merged_verses

    def _merge_verse_continuations(self, verses: List[Dict], lines: List[str]) -> List[Dict]:
        """
        Merge verses that continue across multiple lines.

        In the KJV format, a verse may start on one line and continue on subsequent
        lines until the next verse reference or a blank line.

        Args:
            verses: List of verse dictionaries
            lines: Original lines from the file

        Returns:
            List of merged verse dictionaries
        """
        if not verses:
            return verses

        merged = []

        for i, verse in enumerate(verses):
            # Start with the verse text we already have
            full_text = verse['text']
            current_line = verse['line_num']

            # Look ahead to find continuation lines
            # Continue until we hit the next verse or a blank line
            next_verse_line = verses[i + 1]['line_num'] if i + 1 < len(verses) else len(lines) + 1

            # Check lines between this verse and the next verse
            for line_idx in range(current_line, next_verse_line - 1):
                if line_idx >= len(lines):
                    break

                line = lines[line_idx].strip()

                # Skip the line where the verse starts (we already have that text)
                if line_idx + 1 == current_line:
                    continue

                # Stop at blank lines
                if not line:
                    break

                # Check if this line contains a verse reference (indicating a new verse)
                if re.search(r'\d+:\d+\s+', line):
                    break

                # This is a continuation line - append it
                full_text += ' ' + line

            # Create merged verse entry
            merged_verse = verse.copy()
            merged_verse['text'] = full_text.strip()
            merged.append(merged_verse)

        return merged


def chunk_bible_by_verse(file_path: str) -> List[Tuple[str, int, int, str]]:
    """
    Chunk a Bible file by verse.

    Args:
        file_path: Path to the Bible text file

    Returns:
        List of tuples (verse_text, start_position, line_number, verse_reference)
    """
    parser = BibleParser()
    verses = parser.parse_bible_file(file_path)

    # Convert to the format expected by the embedder
    chunks = []
    for verse in verses:
        chunks.append((
            verse['text'],
            verse['start_pos'],
            verse['line_num'],
            verse['reference']
        ))

    return chunks
