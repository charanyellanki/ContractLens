"""Tests for data module."""

import pytest

from contractlens.data.chunker import Chunk, ContractChunker
from contractlens.data.cuad_loader import CUADLoader


class TestContractChunker:
    """Tests for ContractChunker."""

    def test_chunk_by_paragraph(self):
        """Test paragraph-based chunking."""
        chunker = ContractChunker(chunk_size=1000, overlap=100)
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        
        chunks = chunker.chunk_by_paragraph(text)
        
        assert len(chunks) == 3
        assert chunks[0].text == "Paragraph 1."
        assert chunks[1].text == "Paragraph 2."
        assert chunks[2].text == "Paragraph 3."

    def test_chunk_with_sliding_window(self):
        """Test sliding window chunking."""
        chunker = ContractChunker(chunk_size=10, overlap=2)
        text = "0123456789" * 5  # 50 chars
        
        chunks = chunker.chunk_with_sliding_window(text)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_default_method(self):
        """Test default chunking method."""
        chunker = ContractChunker()
        text = "Test paragraph.\n\nAnother paragraph."
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 2


class TestCUADLoader:
    """Tests for CUADLoader."""

    def test_load_categories(self):
        """Test loading CUAD categories."""
        loader = CUADLoader()
        categories = loader.load_categories()
        
        assert len(categories) == 41
        assert "Confidentiality" in categories
        assert "Termination" in categories
        assert "Indemnification" in categories

    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = CUADLoader(data_path="/tmp/cuad")
        
        assert loader.data_path.name == "cuad"