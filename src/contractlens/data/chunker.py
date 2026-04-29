"""Contract chunking utilities for retrieval."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    """A chunk of text from a contract."""

    text: str
    start_char: int
    end_char: int
    chunk_id: str


class ContractChunker:
    """Chunks contracts for retrieval."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        use_semantic: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_semantic = use_semantic

    def chunk_by_paragraph(self, text: str) -> list[Chunk]:
        """Chunk text by paragraphs."""
        paragraphs = text.split("\n\n")
        chunks: list[Chunk] = []
        current_pos = 0

        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append(
                    Chunk(
                        text=para.strip(),
                        start_char=current_pos,
                        end_char=current_pos + len(para),
                        chunk_id=f"para_{i}",
                    )
                )
            current_pos += len(para) + 2  # +2 for \n\n

        return chunks

    def chunk_with_sliding_window(self, text: str) -> list[Chunk]:
        """Chunk text with sliding window."""
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            chunk_text = text[i : i + self.chunk_size]
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start_char=i,
                        end_char=min(i + self.chunk_size, len(text)),
                        chunk_id=f"window_{i // step}",
                    )
                )

        return chunks

    def chunk(self, text: str, method: Optional[str] = None) -> list[Chunk]:
        """Chunk contract text using specified method."""
        method = method or ("semantic" if self.use_semantic else "paragraph")
        
        if method == "semantic":
            # TODO: Implement semantic chunking using embeddings
            return self.chunk_by_paragraph(text)
        elif method == "sliding_window":
            return self.chunk_with_sliding_window(text)
        else:
            return self.chunk_by_paragraph(text)