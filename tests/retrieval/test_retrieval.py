"""Tests for retrieval module."""

import pytest

from contractlens.retrieval.chroma_store import ChromaStore, RetrievedChunk
from contractlens.retrieval.hybrid_retriever import HybridRetriever, HybridResult
from contractlens.retrieval.reranker import Reranker


class TestChromaStore:
    """Tests for ChromaStore."""

    def test_store_initialization(self):
        """Test store initialization."""
        store = ChromaStore(collection_name="test_contracts")
        
        assert store.collection_name == "test_contracts"

    def test_retrieve_returns_list(self):
        """Test retrieve returns list of chunks."""
        store = ChromaStore()
        result = store.retrieve(
            query_embedding=[0.1] * 384,
            n_results=5,
        )
        
        assert isinstance(result, list)


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_retriever_initialization(self):
        """Test retriever initialization."""
        store = ChromaStore()
        retriever = HybridRetriever(store, alpha=0.5)
        
        assert retriever.alpha == 0.5
        assert retriever.use_reranker is True


class TestReranker:
    """Tests for Reranker."""

    def test_reranker_initialization(self):
        """Test reranker initialization."""
        reranker = Reranker()
        
        assert reranker.model_name is not None

    def test_rerank_empty_candidates(self):
        """Test reranking with empty candidates."""
        reranker = Reranker()
        result = reranker.rerank("query", [], top_k=5)
        
        assert result == []

    def test_score_single_pair(self):
        """Test scoring a single query-document pair."""
        reranker = Reranker()
        score = reranker.score("What is termination?", "The termination clause...")
        
        assert isinstance(score, float)