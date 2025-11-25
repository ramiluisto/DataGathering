"""Text chunking utilities for splitting documents into smaller pieces."""

from typing import Literal, Optional

import tiktoken

from datagather.sources.base import Chunk, Document


def get_encoder(model: str = "cl100k_base") -> tiktoken.Encoding:
    """Get tiktoken encoder.

    Args:
        model: Encoding model name (default: cl100k_base for GPT-4/Claude)

    Returns:
        Tiktoken encoder
    """
    return tiktoken.get_encoding(model)


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in text.

    Args:
        text: Text to count tokens in
        model: Encoding model name

    Returns:
        Number of tokens
    """
    encoder = get_encoder(model)
    return len(encoder.encode(text))


def chunk_by_tokens(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "cl100k_base",
) -> list[dict]:
    """Chunk text by token count with overlap.

    Args:
        text: Text to chunk
        chunk_size: Target tokens per chunk
        overlap: Overlap tokens between chunks
        model: Tiktoken encoding model

    Returns:
        List of chunk dictionaries with text and metadata
    """
    encoder = get_encoder(model)
    tokens = encoder.encode(text)

    if len(tokens) <= chunk_size:
        return [
            {
                "chunk_index": 0,
                "text": text,
                "token_count": len(tokens),
                "char_count": len(text),
                "start_char": 0,
                "end_char": len(text),
            }
        ]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)

        # Calculate character positions
        prefix_tokens = tokens[:start]
        prefix_text = encoder.decode(prefix_tokens) if prefix_tokens else ""
        start_char = len(prefix_text)
        end_char = start_char + len(chunk_text)

        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "char_count": len(chunk_text),
                "start_char": start_char,
                "end_char": end_char,
            }
        )

        chunk_index += 1

        # Move start with overlap
        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def chunk_by_sentences(
    text: str,
    target_tokens: int = 512,
    model: str = "cl100k_base",
) -> list[dict]:
    """Chunk text by sentences, targeting approximate token count.

    Args:
        text: Text to chunk
        target_tokens: Target tokens per chunk
        model: Tiktoken encoding model

    Returns:
        List of chunk dictionaries
    """
    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)
    except ImportError:
        # Fallback to simple sentence splitting
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)

    encoder = get_encoder(model)
    chunks = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0
    current_start_char = 0

    for sentence in sentences:
        sent_tokens = len(encoder.encode(sentence))

        if current_tokens + sent_tokens > target_tokens and current_sentences:
            # Save current chunk
            chunk_text = " ".join(current_sentences)
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "token_count": current_tokens,
                    "char_count": len(chunk_text),
                    "start_char": current_start_char,
                    "end_char": current_start_char + len(chunk_text),
                }
            )
            chunk_index += 1
            current_start_char += len(chunk_text) + 1  # +1 for space

            # Start new chunk
            current_sentences = [sentence]
            current_tokens = sent_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sent_tokens

    # Don't forget the last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": chunk_text,
                "token_count": current_tokens,
                "char_count": len(chunk_text),
                "start_char": current_start_char,
                "end_char": current_start_char + len(chunk_text),
            }
        )

    return chunks


def chunk_by_paragraphs(
    text: str,
    target_tokens: int = 512,
    model: str = "cl100k_base",
) -> list[dict]:
    """Chunk text by paragraphs, targeting approximate token count.

    Args:
        text: Text to chunk
        target_tokens: Target tokens per chunk
        model: Tiktoken encoding model

    Returns:
        List of chunk dictionaries
    """
    # Split on double newlines
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        return [
            {
                "chunk_index": 0,
                "text": text,
                "token_count": count_tokens(text, model),
                "char_count": len(text),
                "start_char": 0,
                "end_char": len(text),
            }
        ]

    encoder = get_encoder(model)
    chunks = []
    current_paragraphs: list[str] = []
    current_tokens = 0
    chunk_index = 0
    current_start_char = 0

    for para in paragraphs:
        para_tokens = len(encoder.encode(para))

        if current_tokens + para_tokens > target_tokens and current_paragraphs:
            # Save current chunk
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "token_count": current_tokens,
                    "char_count": len(chunk_text),
                    "start_char": current_start_char,
                    "end_char": current_start_char + len(chunk_text),
                }
            )
            chunk_index += 1
            current_start_char += len(chunk_text) + 2  # +2 for \n\n

            # Start new chunk
            current_paragraphs = [para]
            current_tokens = para_tokens
        else:
            current_paragraphs.append(para)
            current_tokens += para_tokens

    # Don't forget the last chunk
    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": chunk_text,
                "token_count": current_tokens,
                "char_count": len(chunk_text),
                "start_char": current_start_char,
                "end_char": current_start_char + len(chunk_text),
            }
        )

    return chunks


def chunk_document(
    doc: Document,
    strategy: Literal["token", "sentence", "paragraph"] = "token",
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "cl100k_base",
) -> list[Chunk]:
    """Chunk a document using the specified strategy.

    Args:
        doc: Document to chunk
        strategy: Chunking strategy
        chunk_size: Target tokens per chunk
        overlap: Overlap tokens (for token strategy only)
        model: Tiktoken encoding model

    Returns:
        List of Chunk objects
    """
    if strategy == "token":
        chunk_dicts = chunk_by_tokens(doc.text, chunk_size, overlap, model)
    elif strategy == "sentence":
        chunk_dicts = chunk_by_sentences(doc.text, chunk_size, model)
    elif strategy == "paragraph":
        chunk_dicts = chunk_by_paragraphs(doc.text, chunk_size, model)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    total_chunks = len(chunk_dicts)
    chunks = []

    for chunk_dict in chunk_dicts:
        chunk = Chunk(
            id=f"{doc.id}_chunk_{chunk_dict['chunk_index']:03d}",
            document_id=doc.id,
            source=doc.source,
            chunk_index=chunk_dict["chunk_index"],
            total_chunks=total_chunks,
            text=chunk_dict["text"],
            metadata={
                "token_count": chunk_dict["token_count"],
                "char_count": chunk_dict["char_count"],
                "start_char": chunk_dict["start_char"],
                "end_char": chunk_dict["end_char"],
            },
        )
        chunks.append(chunk)

    return chunks
