"""Text cleaning and normalization utilities."""

import re
import unicodedata
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    - Replace multiple spaces with single space
    - Replace multiple newlines with double newline
    - Strip leading/trailing whitespace

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    # Normalize newlines
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\r", "\n", text)
    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Strip overall
    return text.strip()


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode text.

    Args:
        text: Text to normalize
        form: Unicode normalization form (NFC, NFKC, NFD, NFKD)

    Returns:
        Normalized text
    """
    return unicodedata.normalize(form, text)


def remove_control_chars(text: str, keep_newlines: bool = True) -> str:
    """Remove control characters from text.

    Args:
        text: Text to clean
        keep_newlines: Whether to keep newline characters

    Returns:
        Cleaned text
    """
    if keep_newlines:
        # Remove control chars except newlines and tabs
        return "".join(
            char
            for char in text
            if unicodedata.category(char) != "Cc" or char in "\n\r\t"
        )
    else:
        return "".join(char for char in text if unicodedata.category(char) != "Cc")


def clean_text(
    text: str,
    normalize_ws: bool = True,
    normalize_uni: bool = True,
    remove_ctrl: bool = True,
) -> str:
    """Apply standard text cleaning.

    Args:
        text: Text to clean
        normalize_ws: Normalize whitespace
        normalize_uni: Normalize Unicode (NFKC)
        remove_ctrl: Remove control characters

    Returns:
        Cleaned text
    """
    if remove_ctrl:
        text = remove_control_chars(text)
    if normalize_uni:
        text = normalize_unicode(text)
    if normalize_ws:
        text = normalize_whitespace(text)
    return text


def truncate_text(
    text: str,
    max_chars: Optional[int] = None,
    max_words: Optional[int] = None,
    suffix: str = "...",
) -> str:
    """Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_chars: Maximum characters (including suffix)
        max_words: Maximum words
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if max_words is not None:
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + suffix

    if max_chars is not None:
        if len(text) > max_chars:
            text = text[: max_chars - len(suffix)] + suffix

    return text


def extract_text_from_html(html: str) -> str:
    """Extract plain text from HTML.

    Simple extraction without external dependencies.
    For better results, use beautifulsoup4.

    Args:
        html: HTML string

    Returns:
        Plain text
    """
    # Remove script and style elements
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.I)
    # Remove HTML tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode HTML entities
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"&lt;", "<", html)
    html = re.sub(r"&gt;", ">", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"&quot;", '"', html)
    html = re.sub(r"&#\d+;", "", html)
    # Clean up
    return normalize_whitespace(html)


def is_english(text: str, threshold: float = 0.8) -> bool:
    """Check if text is likely English.

    Simple heuristic based on common English words.

    Args:
        text: Text to check
        threshold: Minimum ratio of recognized words

    Returns:
        True if likely English
    """
    common_words = {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "is",
        "are",
        "was",
        "were",
        "been",
    }

    words = text.lower().split()
    if not words:
        return False

    # Sample words if text is long
    if len(words) > 1000:
        import random

        words = random.sample(words, 1000)

    recognized = sum(1 for word in words if word in common_words)
    return (recognized / len(words)) >= threshold * 0.1  # Adjust for common word frequency


def count_words(text: str) -> int:
    """Count words in text.

    Args:
        text: Text to count

    Returns:
        Word count
    """
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences in text (approximate).

    Args:
        text: Text to count

    Returns:
        Sentence count
    """
    # Simple heuristic: count sentence-ending punctuation
    return len(re.findall(r"[.!?]+", text))


def get_text_stats(text: str) -> dict:
    """Get basic statistics about text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with statistics
    """
    return {
        "char_count": len(text),
        "word_count": count_words(text),
        "sentence_count": count_sentences(text),
        "line_count": text.count("\n") + 1,
        "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
    }
