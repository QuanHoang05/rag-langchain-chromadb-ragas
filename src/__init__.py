"""
Gói nội bộ của hệ thống Advanced RAG.
Xuất các class/function chính để các module khác import gọn hơn.
"""
# Tầng nạp dữ liệu
from src.loaders import SimpleLoader, clean_vietnamese_text

# Tầng chia nhỏ văn bản
from src.chunkers import SemanticChunker, FixedSizeChunker, count_tokens

# Tầng lưu trữ & truy xuất
from src.retrievers import VectorStoreManager, BM25Retriever, HybridRetriever, WebSearchRetriever

# Tầng định tuyến
from src.router import QueryRouter

# Tầng biến đổi câu hỏi
from src.transformers import HyDEQueryTransformer, DecompositionQueryTransformer

# Tầng tái xếp hạng
from src.rerankers import CrossEncoderReranker, MMRReranker

# Tầng tinh chỉnh embedding
from src.finetune_embeddings import EmbeddingFinetuner

# Tầng đánh giá & giám sát
from src.evaluator import RAGLogger, RAGEvaluator

# Tầng tích hợp pipeline
from src.chain import AdvancedRAGChain, FocusedAnswerParser

__all__ = [
    # Loaders
    "SimpleLoader", "clean_vietnamese_text",
    # Chunkers
    "SemanticChunker", "FixedSizeChunker", "count_tokens",
    # Retrievers
    "VectorStoreManager", "BM25Retriever", "HybridRetriever", "WebSearchRetriever",
    # Router
    "QueryRouter",
    # Transformers
    "HyDEQueryTransformer", "DecompositionQueryTransformer",
    # Rerankers
    "CrossEncoderReranker", "MMRReranker",
    # Fine-tuning
    "EmbeddingFinetuner",
    # Evaluator
    "RAGLogger", "RAGEvaluator",
    # Chain
    "AdvancedRAGChain", "FocusedAnswerParser",
]
