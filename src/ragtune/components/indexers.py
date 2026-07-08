"""
Indexer Factory
===============
Provides pluggable retrieval strategies: BM25, dense (FAISS), and hybrid.

Usage:
    indexer = IndexerFactory.create("bm25", num_results=100)
    retriever = indexer.build(corpus)

    # Or via environment:
    INDEX_TYPE=bm25 python scripts/run_benchmark.py
"""

import os
import tempfile
from typing import Dict, Optional


class BaseIndexer:
    """Base class for all indexers."""

    def __init__(self, num_results: int = 100):
        self.num_results = num_results

    def build(self, corpus: Dict[str, Dict]):
        """Build retriever from corpus. Returns a BaseRetriever."""
        raise NotImplementedError


class BM25Indexer(BaseIndexer):
    """PyTerrier BM25 indexer."""

    def __init__(self, index_dir: Optional[str] = None, num_results: int = 100):
        super().__init__(num_results)
        self.index_dir = index_dir

    def build(self, corpus):
        import pyterrier as pt
        from ragtune.adapters.pyterrier import PyTerrierRetriever

        if not pt.java.started():
            pt.java.init()

        corpus_list = [{"docno": did, "text": d["text"]} for did, d in corpus.items()]
        idx_dir = self.index_dir or os.path.join(tempfile.mkdtemp(), "idx")
        indexer = pt.IterDictIndexer(
            idx_dir, overwrite=True, meta={"docno": 128, "text": 4096}
        )
        index_ref = indexer.index(iter(corpus_list))
        bm25 = pt.terrier.Retriever(
            index_ref,
            wmodel="BM25",
            metadata=["docno", "text"],
            num_results=self.num_results,
        )
        return PyTerrierRetriever(bm25)


class DenseIndexer(BaseIndexer):
    """FAISS dense indexer using HuggingFace embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", num_results: int = 100):
        super().__init__(num_results)
        self.model_name = model_name

    def build(self, corpus):
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from ragtune.adapters.langchain import LangChainRetriever

        docs = [
            Document(page_content=d["text"], metadata={"id": did})
            for did, d in corpus.items()
        ]
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        vectorstore = FAISS.from_documents(docs, embeddings)
        return LangChainRetriever(
            vectorstore.as_retriever(search_kwargs={"k": self.num_results})
        )


class IndexerFactory:
    """Factory for creating retrieval indexers."""

    _REGISTRY = {
        "bm25": BM25Indexer,
        "dense": DenseIndexer,
    }

    @classmethod
    def register(cls, name: str, indexer_cls):
        """Register a new indexer type."""
        cls._REGISTRY[name] = indexer_cls

    @classmethod
    def create(cls, index_type: str = "bm25", **kwargs) -> BaseIndexer:
        """Create an indexer by type name.

        Args:
            index_type: "bm25", "dense", or custom registered name
            **kwargs: Passed to the indexer constructor

        Returns:
            BaseIndexer instance
        """
        indexer_cls = cls._REGISTRY.get(index_type)
        if indexer_cls is None:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown indexer type: {index_type!r}. Available: {available}"
            )
        return indexer_cls(**kwargs)

    @classmethod
    def from_env(cls) -> BaseIndexer:
        """Create indexer from INDEX_TYPE env var."""
        index_type = os.environ.get("INDEX_TYPE", "bm25")
        num_results = int(os.environ.get("TOP_K", "100"))
        kwargs = {"num_results": num_results}
        if index_type == "dense":
            kwargs["model_name"] = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return cls.create(index_type, **kwargs)
