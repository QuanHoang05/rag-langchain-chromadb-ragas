import re
import time
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel

# Import các thành phần nội bộ hệ thống
from src.router import QueryRouter
from src.retrievers import HybridRetriever, WebSearchRetriever, BM25Retriever
from src.transformers import HyDEQueryTransformer, DecompositionQueryTransformer
from src.rerankers import CrossEncoderReranker, MMRReranker
from src.evaluator import RAGLogger

class FocusedAnswerParser:
    """
    Giao diện Giai đoạn 4: Hậu xử lý phản hồi (Post-processing).
    - Cắt lọc các cụm từ dẫn dắt lặp lại ("Dựa vào tài liệu...", "Câu trả lời là...").
    - Làm sạch định dạng thụt lề, ngắt dòng thừa để hiển thị gọn gàng trên UI.
    """
    # Các pattern dùng lại để tránh re-compile lặp lại
    _BULLET_PATTERN = re.compile(r"^\s*[\u2022\-\*]\s*", re.MULTILINE)
    _NEWLINE_PATTERN = re.compile(r"\n+")

    @staticmethod
    def parse(text: str) -> str:
        """
        Hậu xử lý phản hồi LLM:
        - Cắt bỏ tiền tố dẫn dắt do prompt template tạo ra.
        - Loại bỏ ký hiệu danh sách thừa ở đầu dòng.
        - Gộp xuống dòng thành khoảng trắng đơn để hiển thị sạch trên UI.
        """
        if not text:
            return ""
        text = text.strip()
        
        # Bug fix: Tách bằng split chỉ khi nhãn thực sự tồn tại trong phản hồi
        if "[TRẢ LỜI]:" in text:
            text = text.split("[TRẢ LỜI]:")[-1].strip()
        elif "Trả lời:" in text:
            text = text.split("Trả lời:")[-1].strip()
            
        # Loại bỏ các dấu dòng liệt kê thừa ở đầu câu (vd: -, *, •)
        text = FocusedAnswerParser._BULLET_PATTERN.sub("", text)
        
        # Gộp các ký tự xuống dòng thành khoảng trắng đơn
        text = FocusedAnswerParser._NEWLINE_PATTERN.sub(" ", text).strip()
        return text

class AdvancedRAGChain:
    """
    Hệ thống kết nối chuỗi RAG nâng cao (Advanced RAG Pipeline Chain).
    - Đồng bộ hóa định tuyến, biến đổi truy vấn, tìm kiếm lai, nén ngữ cảnh và sinh câu trả lời.
    - Áp dụng triệt để cơ chế tối ưu hóa Token 4 giai đoạn.
    - BM25 index được cache và làm mới (rebuild) chỉ khi có tài liệu mới được nạp.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
        embeddings: Any,
        vector_manager: Any,
        log_dir: str = "./logs"
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_manager = vector_manager
        
        # Khởi tạo các module bổ trợ
        self.router = QueryRouter(llm)
        self.hyde_transformer = HyDEQueryTransformer(llm)
        self.decomp_transformer = DecompositionQueryTransformer(llm)
        
        # Rerankers (Cấu hình mặc định giữ lại tối đa 3-5 tài liệu tốt nhất - GD 3 Context Compression)
        self.cross_encoder_reranker = CrossEncoderReranker(top_k=3)
        self.mmr_reranker = MMRReranker(embedding_model=embeddings, top_k=3)
        
        # Web Search
        self.web_search = WebSearchRetriever(max_results=5)
        
        # Logger
        self.logger = RAGLogger(log_dir)
        
        # Cache BM25 index để tránh rebuild tốn kém trên mỗi request
        # _bm25_cache_key: số lượng document trong DB dùng để phát hiện khi nào cần rebuild
        self._bm25_retriever_cache: Optional[BM25Retriever] = None
        self._bm25_cache_doc_count: int = 0
        
        # Template Prompt sinh câu trả lời (GD 4: Giới hạn tokens bằng Prompt cấu trúc gắt gao)
        self.prompt_template = """Bạn là trợ lý AI chuyên nghiệp phân tích tài liệu tiếng Việt.
Hãy trả lời câu hỏi dựa trên các tài liệu được cung cấp dưới đây.

[TÀI LIỆU]:
{context}

[CÂU HỎI]:
{question}

[YÊU CẦU]:
- Trả lời trung thực, ngắn gọn và trực tiếp vào câu hỏi trong khoảng 3 đến 5 câu chi tiết.
- Không lặp lại câu hỏi, không chào hỏi dông dài, không tự tạo thông tin ngoài tài liệu được cung cấp.
- Nếu tài liệu không chứa câu trả lời, hãy nói rõ: "Không có thông tin trong tài liệu".

[TRẢ LỜI]:"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Gộp nội dung các chunks thành chuỗi văn bản context để nạp vào prompt.
        Loại bỏ các chunk bị trùng lặp hoặc quá ngắn (< 40 ký tự).
        """
        formatted = []
        seen = set()
        for doc in docs:
            content = doc.page_content.strip()
            if content and len(content) > 40 and content not in seen:
                formatted.append(content)
                seen.add(content)
        return "\n\n".join(formatted)

    def run(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Luồng thực thi RAG Chain đồng bộ.
        """
        start_time = time.time()
        execution_logs = {}
        
        # Đọc tham số cấu hình từ UI Gradio
        search_mode = config.get("search_mode", "hybrid")  # 'dense', 'sparse', 'hybrid'
        fusion_mode = config.get("fusion_mode", "rrf")      # 'rrf', 'interleave'
        query_transform = config.get("query_transform", "none") # 'none', 'hyde', 'decomposition'
        reranker_type = config.get("reranker_type", "none")  # 'none', 'cross_encoder', 'mmr'
        use_router = config.get("use_router", True)
        force_internet = config.get("force_internet", False) # Ép buộc tra cứu web
        selected_tags = config.get("doc_tags", [])          # Bộ lọc tags (Pre-filtering)
        
        # Bước 1: Định tuyến truy vấn (Query Routing)
        route = "local"
        if use_router and not force_internet:
            route = self.router.route_query(query)
            print(f"[Router] Định tuyến câu hỏi '{query}' sang kênh: '{route}'")
            execution_logs["route_decision"] = route
        elif force_internet:
            route = "web"
            execution_logs["route_decision"] = "web (forced)"
            
        # Bước 2: Biến đổi truy vấn (Query Transformation) và lấy tài liệu
        retrieved_documents = []
        
        # Khởi tạo retriever dựa trên các tài liệu đã nạp trong vector manager
        vector_retriever = self.vector_manager.get_retriever(k=10, doc_tags=selected_tags)
        
        # Khởi tạo BM25 Retriever: lấy dữ liệu từ VectorDB một lần duy nhất (tránh double round-trip)
        # Cache BM25 và chỉ rebuild khi phát hiện số lượng document trong DB thay đổi
        bm25_retriever = None
        try:
            db_data = self.vector_manager.db.get()  # Chỉ gọi 1 lần
            raw_docs = db_data["documents"]
            metadatas = db_data["metadatas"]
            current_doc_count = len(raw_docs)
            
            # Phát hiện thay đổi: Nếu số document thay đổi hoặc chưa có cache thì rebuild BM25
            if (
                self._bm25_retriever_cache is None 
                or self._bm25_cache_doc_count != current_doc_count
            ):
                all_docs_for_bm25 = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(raw_docs, metadatas)
                    if text.strip()  # Bỏ qua document rỗng
                ]
                if all_docs_for_bm25:
                    self._bm25_retriever_cache = BM25Retriever.from_documents(all_docs_for_bm25, k=10)
                    self._bm25_cache_doc_count = current_doc_count
                    print(f"[BM25] Rebuilt index với {current_doc_count} documents.")
            
            # Lọc tài liệu theo tag cho BM25 (Post-filter trên kết quả trả về)
            # Lưu ý: BM25 không hỗ trợ pre-filter như Chroma, ta filter sau kết quả truy xuất
            bm25_retriever = self._bm25_retriever_cache
            
        except Exception as e:
            print(f"[Cảnh báo] Lỗi khởi tạo BM25: {str(e)}")
            bm25_retriever = None

        # Thực thi theo kênh định tuyến
        if route == "web":
            # Chỉ lấy tài liệu từ web
            retrieved_documents = self.web_search.invoke(query)
            execution_logs["retrieval_source"] = "DuckDuckGo Web Search"
            
        else:
            # Luồng tìm kiếm tài liệu Local (có thể kết hợp Web nếu là 'hybrid')
            search_queries = [query]
            
            # Áp dụng Biến đổi truy vấn
            if query_transform == "hyde":
                hyde_doc = self.hyde_transformer.generate_hypothetical_doc(query)
                search_queries = [hyde_doc]
                execution_logs["hyde_document"] = hyde_doc
                print(f"[Transformer] Đã tạo văn bản giả định HyDE: {hyde_doc[:80]}...")
            elif query_transform == "decomposition":
                sub_queries = self.decomp_transformer.decompose(query)
                search_queries = sub_queries + [query] # Gộp câu hỏi phụ và câu hỏi gốc
                execution_logs["decomposed_queries"] = sub_queries
                print(f"[Transformer] Đã phân rã thành {len(sub_queries)} câu hỏi phụ: {sub_queries}")

            # Truy xuất tài liệu cho từng câu hỏi tìm kiếm
            local_docs = []
            seen_content = set()
            
            for q in search_queries:
                q_docs = []
                if search_mode == "dense":
                    q_docs = vector_retriever.invoke(q)
                elif search_mode == "sparse" and bm25_retriever:
                    q_docs = bm25_retriever.invoke(q)
                elif search_mode == "hybrid" and bm25_retriever:
                    hybrid_engine = HybridRetriever(
                        bm25_retriever=bm25_retriever,
                        vector_retriever=vector_retriever,
                        top_k=10,
                        fusion_mode=fusion_mode
                    )
                    q_docs = hybrid_engine.invoke(q)
                else:
                    # Fallback
                    q_docs = vector_retriever.invoke(q)
                    
                # Gộp tài liệu loại bỏ trùng lặp nội dung
                for doc in q_docs:
                    if doc.page_content not in seen_content:
                        local_docs.append(doc)
                        seen_content.add(doc.page_content)
                        
            retrieved_documents = local_docs
            execution_logs["retrieval_source"] = "Local Knowledge Base"
            
            # Nếu định tuyến là 'hybrid' (kết hợp) hoặc không tìm thấy tài liệu local nào
            if route == "hybrid" or (len(retrieved_documents) == 0 and use_router):
                print("[Retriever] Luồng kết hợp được kích hoạt. Tiến hành bổ sung tài liệu Web...")
                web_docs = self.web_search.invoke(query)
                # Gộp tài liệu Web vào danh sách
                for doc in web_docs:
                    if doc.page_content not in seen_content:
                        retrieved_documents.append(doc)
                        seen_content.add(doc.page_content)
                execution_logs["retrieval_source"] += " + Web Search"

        # Bước 3: Nén ngữ cảnh & Reranking (GD 3: Context Compression)
        reranked_documents = retrieved_documents
        
        if reranker_type == "cross_encoder":
            # Sử dụng Cross-Encoder cục bộ để tái xếp hạng
            reranked_documents = self.cross_encoder_reranker.rerank(query, retrieved_documents)
        elif reranker_type == "mmr":
            # Sử dụng MMR phi tham số để đa dạng hóa
            reranked_documents = self.mmr_reranker.rerank(query, retrieved_documents)
        else:
            # Nếu không rerank, cắt tỉa cứng lấy tối đa top 3 tài liệu để bảo vệ token
            reranked_documents = retrieved_documents[:3]
            
        execution_logs["raw_retrieved_count"] = len(retrieved_documents)
        execution_logs["reranked_count"] = len(reranked_documents)

        # Bước 4: Tạo Prompt và gọi LLM Sinh phản hồi (GD 4: Generation)
        context_str = self._format_docs(reranked_documents)
        
        # Nếu hoàn toàn không có dữ liệu ngữ cảnh
        if not context_str.strip():
            context_str = "Không có tài liệu phù hợp được tìm thấy."
            
        formatted_prompt = self.prompt.format(
            context=context_str,
            question=query
        )
        
        # Sinh câu trả lời qua LLM (Gemini / OpenAI / Qwen)
        try:
            # Ép trần Output token
            response = self.llm.invoke(formatted_prompt)
            raw_answer = response.content.strip()
        except Exception as e:
            raw_answer = f"[Lỗi hệ thống sinh câu trả lời]: {str(e)}"
            
        # Hậu xử lý phản hồi bằng FocusedAnswerParser
        final_answer = FocusedAnswerParser.parse(raw_answer)
        
        # Trích xuất danh sách nguồn trích dẫn
        citations = []
        for doc in reranked_documents:
            meta = doc.metadata
            citations.append({
                "source": meta.get("source_name", "Tài liệu không rõ nguồn"),
                "tag": meta.get("doc_tag", "Chưa phân loại"),
                "page": meta.get("page", 1),
                "snippet": doc.page_content[:150] + "..."
            })
            
        elapsed_time = time.time() - start_time
        
        # Ghi nhận vào trace log phục vụ quan sát và đánh giá RAGAS sau này
        self.logger.log_trace(
            query=query,
            answer=final_answer,
            contexts=[doc.page_content for doc in reranked_documents],
            metadata_sources=citations,
            execution_time=elapsed_time,
            route=route
        )
        
        return {
            "query": query,
            "answer": final_answer,
            "citations": citations,
            "execution_time_seconds": elapsed_time,
            "logs": execution_logs,
            "context_used": context_str
        }
