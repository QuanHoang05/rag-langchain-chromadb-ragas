import numpy as np
import tiktoken
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from underthesea import sent_tokenize  # Tokenizer câu chuyên dụng cho Tiếng Việt

# Khởi tạo bộ đếm token bằng tiktoken (sử dụng cl100k_base làm chuẩn)
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tokenizer = None

def count_tokens(text: str) -> int:
    """
    Hàm tính toán chính xác số lượng tokens trong chuỗi văn bản.
    """
    if not text:
        return 0
    if _tokenizer:
        return len(_tokenizer.encode(text, disallowed_special=()))
    # Fallback dự phòng nếu không tải được bộ mã hóa tiktoken (tính bằng số từ ước lượng)
    return len(text.split())

class FixedSizeChunker:
    """
    Bộ chia nhỏ văn bản theo độ dài cố định sử dụng RecursiveCharacterTextSplitter.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

class SemanticChunker:
    """
    Giai đoạn 1 (GD 1: Ingestion): Bộ chia nhỏ văn bản dựa trên ngữ nghĩa (Semantic Chunker).
    - Cắt câu tiếng Việt bằng Underthesea.
    - So sánh cosine similarity của các câu kề nhau bằng embedding model.
    - Giới hạn cứng số token bằng tiktoken (Max 300-400 tokens/chunk) để bảo vệ Input Tokens của LLM.
    """
    def __init__(
        self, 
        embedding_model: Any, 
        breakpoint_threshold: float = 0.5,
        min_chunk_tokens: int = 100,
        max_chunk_tokens: int = 350,
        overlap_chars: int = 128     # Số ký tự gối đầu (tính bằng ký tự, nhất quán hơn token)
    ):
        self.embeddings = embedding_model
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_chars = overlap_chars

    def _calculate_cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Tính khoảng cách cosine giữa hai vector embedding.
        """
        v1, v2 = np.array(emb1), np.array(emb2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Tách văn bản thành danh sách các câu tiếng Việt sạch sẽ.
        """
        sentences = sent_tokenize(text)
        # Bỏ qua các câu quá ngắn không chứa đủ thông tin ngữ nghĩa
        return [s.strip() for s in sentences if s.strip() and len(s) > 15]

    def _chunk_by_semantic_similarity(self, sentences: List[str]) -> List[str]:
        """
        Gom nhóm các câu thành các chunk dựa trên độ tương đồng cosine và giới hạn số token.
        """
        if not sentences:
            return []

        # Goi embed_documents() MOT LAN duy nhat cho toan bo cac cau trong van ban.
        # LY DO KHONG tu chia batch thu cong:
        #   1. Semantic Chunking can tinh cosine similarity LIEN TUC giua cac cau ke nhau.
        #      Neu chia batch, ranh gioi giua batch 1 va batch 2 mat tinh lien tuc -> diem cat sai.
        #   2. Doi voi API (Gemini / OpenAI): Rate limit du lon cho 2000+ cau ngan, khong bi timeout.
        #   3. Doi voi model local (sentence-transformers): thu vien da tu dong chia batch ngam
        #      ben trong ham encode(), khong can ta lam them gi nua.
        try:
            sentence_embeddings = self.embeddings.embed_documents(sentences)
        except Exception as e:
            print(f"[Cảnh báo] Lỗi gọi embedding cho chunking: {str(e)}. Fallback sang gộp cố định.")
            # Fallback nếu lỗi embedding (gộp tạm 5 câu thành 1 chunk)
            chunks = []
            for i in range(0, len(sentences), 5):
                chunks.append(" ".join(sentences[i:i+5]))
            return chunks

        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            prev_emb = sentence_embeddings[i-1]
            curr_emb = sentence_embeddings[i]
            similarity = self._calculate_cosine_similarity(prev_emb, curr_emb)
            
            chunk_text = " ".join(current_chunk)
            chunk_tokens = count_tokens(chunk_text)
            next_sentence_tokens = count_tokens(sentences[i])
            
            # ĐIỀU KIỆN 1: Nếu vượt quá giới hạn trần (max_chunk_tokens), bắt buộc ngắt chunk
            if chunk_tokens + next_sentence_tokens >= self.max_chunk_tokens:
                chunks.append(chunk_text)
                current_chunk = [sentences[i]]
                
            # ĐIỀU KIỆN 2: Nếu chưa đạt giới hạn sàn (min_chunk_tokens), gộp tiếp bất kể tương đồng
            elif chunk_tokens < self.min_chunk_tokens:
                current_chunk.append(sentences[i])
                
            # ĐIỀU KIỆN 3: Đã đủ kích thước tối thiểu, kiểm tra tương đồng ngữ nghĩa
            # Nếu similarity >= breakpoint_threshold -> cùng chủ đề -> gộp tiếp
            elif similarity >= self.breakpoint_threshold:
                current_chunk.append(sentences[i])
                
            # ĐIỀU KIỆN 4: Chuyển chủ đề ngữ nghĩa -> ngắt chunk, tạo nhóm mới
            else:
                chunks.append(chunk_text)
                current_chunk = [sentences[i]]
                
        # Thêm nhóm câu cuối cùng còn sót lại
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Thực thi phân tách danh sách tài liệu thành các chunks ngữ nghĩa kèm theo gối đầu và dán nhãn.
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.page_content
            if not content.strip():
                continue
                
            sentences = self._split_into_sentences(content)
            if not sentences:
                continue
                
            # Gom câu thành các đoạn văn ngữ nghĩa
            chunks_text = self._chunk_by_semantic_similarity(sentences)
            total_chunks = len(chunks_text)
            
            # Xử lý gối đầu ngữ cảnh (Overlap Handling) và đóng gói vào Document
            for idx, chunk_text in enumerate(chunks_text):
                if not chunk_text.strip():
                    continue
                    
                # Gối đầu ngữ cảnh từ chunk trước (tính bằng ký tự, nhất quán với overlap_chars)
                if idx > 0 and self.overlap_chars > 0:
                    prev_chunk = chunks_text[idx - 1]
                    # Chỉ gối đầu nếu chunk trước dài hơn số ký tự overlap
                    if len(prev_chunk) > self.overlap_chars:
                        overlap = prev_chunk[-self.overlap_chars:]
                        chunk_text = overlap + " " + chunk_text
                
                # Copy toàn bộ metadata gốc (như doc_tag, source_name) sang chunk mới
                # Bổ sung chunk_index để dễ tra vết và hiển thị trích dẫn
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["total_chunks"] = total_chunks
                
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk_doc)
                
        return all_chunks
