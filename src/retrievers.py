import os
import hashlib
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize  # Tách từ tiếng Việt chuẩn xác
from duckduckgo_search import DDGS  # Tra cứu web miễn phí không cần API key

class VectorStoreManager:
    """
    Quản lý Vector Database sử dụng ChromaDB.
    Hỗ trợ dán nhãn metadata phục vụ tính năng Pre-filtering (Lọc trước khi tìm kiếm).
    """
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: Any = None):
        self.persist_directory = persist_directory
        self.embeddings = embedding_model
        self.db = None
        
    def init_db(self, documents: List[Document] = None):
        """
        Khởi tạo hoặc nạp lại Vector Database đã có từ thư mục persist.
        """
        if documents and len(documents) > 0:
            print(f"Khởi tạo VectorDB mới tại '{self.persist_directory}' với {len(documents)} chunks...")
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="advanced_rag_collection"
            )
        else:
            print(f"Nạp VectorDB đã có từ thư mục '{self.persist_directory}'...")
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="advanced_rag_collection"
            )
            
    def get_retriever(self, k: int = 5, doc_tags: List[str] = None) -> BaseRetriever:
        """
        Tạo đối tượng Retriever từ VectorDB.
        Hỗ trợ Pre-filtering: chỉ tìm kiếm các chunks có tag nằm trong danh sách doc_tags.
        """
        if not self.db:
            raise ValueError("VectorDB chưa được khởi tạo. Hãy gọi init_db trước.")
            
        search_kwargs = {"k": k}
        
        # Áp dụng bộ lọc trước (Pre-filtering) theo tag
        if doc_tags and len(doc_tags) > 0:
            if len(doc_tags) == 1:
                search_kwargs["filter"] = {"doc_tag": doc_tags[0]}
            else:
                # Cấu hình Chroma filter $in cho danh sách tag
                search_kwargs["filter"] = {"doc_tag": {"$in": doc_tags}}
                
        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )


class BM25Retriever(BaseRetriever):
    """
    Retriever tìm kiếm từ khóa sử dụng thuật toán BM25.
    Đã được Việt hóa nhờ thư viện word_tokenize của Underthesea để tách từ ghép tiếng Việt.
    """
    documents: List[Document]
    bm25: Any
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(cls, documents: List[Document], k: int = 5):
        # Tách từ cho toàn bộ tài liệu
        tokenized_docs = [
            word_tokenize(doc.page_content.lower())
            for doc in documents
        ]
        bm25 = BM25Okapi(tokenized_docs)
        return cls(documents=documents, bm25=bm25, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # Tách từ cho câu hỏi truy vấn
        tokenized_query = word_tokenize(query.lower())
        # Tính điểm BM25 cho từng tài liệu
        scores = self.bm25.get_scores(tokenized_query)
        # Sắp xếp và lấy ra top k chỉ số
        import numpy as np
        top_k_indices = np.argsort(scores)[::-1][:self.k]
        
        # Chỉ trả về các tài liệu có điểm số lớn hơn 0 để lọc nhiễu
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:
                results.append(self.documents[idx])
        return results


class HybridRetriever(BaseRetriever):
    """
    Giai đoạn 3 (GD 3: Context Compression - Tìm kiếm lai):
    - Kết hợp kết quả từ BM25 (Sparse) và Chroma VectorDB (Dense).
    - Hỗ trợ gộp kết quả theo 2 phương pháp:
      1. 'interleave': Đan xen xen kẽ kết quả (Top 1 BM25, Top 1 Vector, Top 2 BM25...).
      2. 'rrf': Reciprocal Rank Fusion (Hợp nhất theo thứ hạng nghịch đảo) giúp xếp hạng thông minh hơn.
    """
    bm25_retriever: BM25Retriever
    vector_retriever: BaseRetriever
    top_k: int = 5
    fusion_mode: str = "rrf"  # 'rrf' hoặc 'interleave'
    rrf_k: int = 60           # Hằng số làm mượt cho thuật toán RRF

    class Config:
        arbitrary_types_allowed = True

    def _get_doc_id(self, doc: Document) -> str:
        """
        Tạo ID duy nhất cho tài liệu bằng cách hash nội dung.
        Tránh dùng raw page_content làm key dict để tiết kiệm bộ nhớ và tránh collision.
        """
        return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

    def _reciprocal_rank_fusion(self, bm25_docs: List[Document], vector_docs: List[Document]) -> List[Document]:
        """
        Thuật toán Reciprocal Rank Fusion (RRF) để xếp hạng lại tài liệu từ hai nguồn.
        """
        from collections import defaultdict
        
        rrf_scores = defaultdict(float)
        doc_map = {}
        
        # Tính điểm cho danh sách BM25
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_id = self._get_doc_id(doc)
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)
            doc_map[doc_id] = doc
            
        # Tính điểm cho danh sách Vector Search
        for rank, doc in enumerate(vector_docs, start=1):
            doc_id = self._get_doc_id(doc)
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)
            doc_map[doc_id] = doc
            
        # Sắp xếp giảm dần theo điểm RRF
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Trả về danh sách đối tượng Document đã sắp xếp
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def _interleave_results(self, bm25_docs: List[Document], vector_docs: List[Document]) -> List[Document]:
        """
        Trộn xen kẽ hai danh sách tài liệu và loại bỏ trùng lặp.
        Thứ tự ưu tiên: Top 1 BM25 -> Top 1 Vector -> Top 2 BM25 -> Top 2 Vector...
        """
        merged = []
        seen = set()
        
        max_len = max(len(bm25_docs), len(vector_docs))
        for i in range(max_len):
            # Lấy BM25 trước theo tài liệu hướng dẫn
            if i < len(bm25_docs):
                doc = bm25_docs[i]
                if doc.page_content not in seen:
                    merged.append(doc)
                    seen.add(doc.page_content)
                if len(merged) >= self.top_k:
                    break
                    
            # Lấy Vector sau
            if i < len(vector_docs):
                doc = vector_docs[i]
                if doc.page_content not in seen:
                    merged.append(doc)
                    seen.add(doc.page_content)
                if len(merged) >= self.top_k:
                    break
                    
        return merged

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # BM25 và Vector Search chạy tuần tự (sequential); có thể nâng lên async song song sau này
        bm25_results = self.bm25_retriever.invoke(query)
        vector_results = self.vector_retriever.invoke(query)
        
        if self.fusion_mode == "rrf":
            fused = self._reciprocal_rank_fusion(bm25_results, vector_results)
        else:
            fused = self._interleave_results(bm25_results, vector_results)
            
        # Trả về top k tài liệu gộp
        return fused[:self.top_k]


class WebSearchRetriever(BaseRetriever):
    """
    Retriever tra cứu thông tin trực tuyến sử dụng DuckDuckGo Search API (miễn phí).
    Tự động biến đổi kết quả web thành dạng Document có dán nhãn tag='Internet'.
    """
    max_results: int = 5
    timeout_seconds: int = 10  # Thời gian chờ tối đa cho một request web

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        print(f"[Internet Search] Đang tra cứu thông tin web cho: '{query}'...")
        web_docs = []
        
        try:
            with DDGS() as ddgs:
                # Cấu hình timeout tránh treo vô thời hạn khi mạng chậm
                results = list(ddgs.text(
                    query, 
                    max_results=self.max_results,
                ))
                
            for idx, r in enumerate(results):
                title = r.get("title", "Tin tức Web")
                link = r.get("href", "http://duckduckgo.com")
                snippet = r.get("body", "")
                
                if snippet and snippet.strip():
                    # Cắt tên file hiển thị cho gọn nếu title quá dài
                    display_title = title[:50] + "..." if len(title) > 50 else title
                    doc = Document(
                        page_content=snippet,
                        metadata={
                            "source": link,
                            "source_name": f"Web: {display_title}",
                            "title": title,
                            "page": 1,
                            "doc_tag": "Internet"
                        }
                    )
                    web_docs.append(doc)
        except Exception as e:
            print(f"[Internet Search] Lỗi khi gọi DuckDuckGo: {str(e)}")
            
        return web_docs
