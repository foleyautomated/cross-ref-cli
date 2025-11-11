"""
Bible verse parser for extracting individual verses with proper references.
"""

import re
from typing import List, Tuple, Dict


class BibleParser:
    """Parses Bible text files and extracts verses with proper book/chapter/verse references."""

    # Hardcoded list of KJV book names (scanned from actual kjv.txt)
    BOOK_NAMES = [
        ("The First Book of Moses: Called Genesis", "Genesis"),
        ("The Second Book of Moses: Called Exodus", "Exodus"),
        ("The Third Book of Moses: Called Leviticus", "Leviticus"),
        ("The Fourth Book of Moses: Called Numbers", "Numbers"),
        ("The Fifth Book of Moses: Called Deuteronomy", "Deuteronomy"),
        ("The Book of Joshua", "Joshua"),
        ("The Book of Judges", "Judges"),
        ("The Book of Ruth", "Ruth"),
        ("The First Book of Samuel", "1 Samuel"),
        ("The Second Book of Samuel", "2 Samuel"),
        ("The First Book of the Kings", "1 Kings"),
        ("The Second Book of the Kings", "2 Kings"),
        ("The First Book of the Chronicles", "1 Chronicles"),
        ("The Second Book of the Chronicles", "2 Chronicles"),
        ("Ezra", "Ezra"),
        ("The Book of Nehemiah", "Nehemiah"),
        ("The Book of Esther", "Esther"),
        ("The Book of Job", "Job"),
        ("The Book of Psalms", "Psalms"),
        ("The Proverbs", "Proverbs"),
        ("Ecclesiastes", "Ecclesiastes"),
        ("The Song of Solomon", "Song of Solomon"),
        ("The Book of the Prophet Isaiah", "Isaiah"),
        ("The Book of the Prophet Jeremiah", "Jeremiah"),
        ("The Lamentations of Jeremiah", "Lamentations"),
        ("The Book of the Prophet Ezekiel", "Ezekiel"),
        ("The Book of Daniel", "Daniel"),
        ("Hosea", "Hosea"),
        ("Joel", "Joel"),
        ("Amos", "Amos"),
        ("Obadiah", "Obadiah"),
        ("Jonah", "Jonah"),
        ("Micah", "Micah"),
        ("Nahum", "Nahum"),
        ("Habakkuk", "Habakkuk"),
        ("Zephaniah", "Zephaniah"),
        ("Haggai", "Haggai"),
        ("Zechariah", "Zechariah"),
        ("Malachi", "Malachi"),
        ("The Gospel According to Saint Matthew", "Matthew"),
        ("The Gospel According to Saint Mark", "Mark"),
        ("The Gospel According to Saint Luke", "Luke"),
        ("The Gospel According to Saint John", "John"),
        ("The Acts of the Apostles", "Acts"),
        ("The Epistle of Paul the Apostle to the Romans", "Romans"),
        ("The First Epistle of Paul the Apostle to the Corinthians", "1 Corinthians"),
        ("The Second Epistle of Paul the Apostle to the Corinthians", "2 Corinthians"),
        ("The Epistle of Paul the Apostle to the Galatians", "Galatians"),
        ("The Epistle of Paul the Apostle to the Ephesians", "Ephesians"),
        ("The Epistle of Paul the Apostle to the Philippians", "Philippians"),
        ("The Epistle of Paul the Apostle to the Colossians", "Colossians"),
        ("The First Epistle of Paul the Apostle to the Thessalonians", "1 Thessalonians"),
        ("The Second Epistle of Paul the Apostle to the Thessalonians", "2 Thessalonians"),
        ("The First Epistle of Paul the Apostle to Timothy", "1 Timothy"),
        ("The Second Epistle of Paul the Apostle to Timothy", "2 Timothy"),
        ("The Epistle of Paul the Apostle to Titus", "Titus"),
        ("The Epistle of Paul the Apostle to Philemon", "Philemon"),
        ("The Epistle of Paul the Apostle to the Hebrews", "Hebrews"),
        ("The General Epistle of James", "James"),
        ("The First Epistle General of Peter", "1 Peter"),
        ("The Second General Epistle of Peter", "2 Peter"),
        ("The First Epistle General of John", "1 John"),
        ("The Second Epistle General of John", "2 John"),
        ("The Third Epistle General of John", "3 John"),
        ("The General Epistle of Jude", "Jude"),
        ("The Revelation of Saint John the Divine", "Revelation"),
    ]

    def __init__(self):
        """Initialize the Bible parser."""
        self.book_map = {full: short for full, short in self.BOOK_NAMES}
        self.current_book = None

    def is_book_title(self, line: str) -> str:
        """
        Check if a line is a book title and return the book name.

        Args:
            line: Line of text to check

        Returns:
            Book name if found, None otherwise
        """
        line_stripped = line.strip()

        # Check against hardcoded book names
        if line_stripped in self.book_map:
            return self.book_map[line_stripped]

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
            book_name = self.is_book_title(line_stripped)
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
