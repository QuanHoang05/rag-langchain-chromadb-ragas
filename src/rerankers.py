import numpy as np
from typing import List, Any
from langchain_core.documents import Document

# Thư viện mô hình học sâu (chỉ sử dụng cho Cross-Encoder cục bộ)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None

class CrossEncoderReranker:
    """
    Giai đoạn 3 (GD 3: Context Compression - Bộ lọc vàng):
    - Tái xếp hạng các tài liệu ứng viên dựa trên mô hình Cross-Encoder (Qwen/Qwen3-Reranker-0.6B).
    - Đo độ liên quan sâu giữa câu hỏi và từng chunk.
    - Cắt tỉa (Slice) danh sách chỉ giữ lại Top k (3-5 chunks) tốt nhất giúp nén Context, tiết kiệm tokens.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", top_k: int = 3):
        self.top_k = top_k
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """
        Nạp mô hình Cross-Encoder từ HuggingFace (Chỉ chạy khi bắt đầu truy vấn để tránh tốn RAM lúc khởi động).
        """
        if self.model is not None:
            return
            
        if torch is None:
            print("[Cảnh báo] Chưa cài đặt transformers/torch. Không thể chạy Cross-Encoder.")
            return

        print(f"Đang nạp mô hình Cross-Encoder Reranker: {self.model_name} trên thiết bị {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            
            # Đảm bảo thiết lập pad_token tránh lỗi pad_token_id is not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).eval().to(self.device)
            
            # Lấy ID của token "yes" và "no" phục vụ chấm điểm xác suất
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.max_length = 8192
            
            # Cấu hình prefix/suffix theo template chuẩn của Qwen Reranker
            self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            
            self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
            print("Đã nạp Cross-Encoder thành công!")
        except Exception as e:
            print(f"[Lỗi] Không thể nạp mô hình reranker {self.model_name}: {str(e)}")
            self.model = None
            
    def _format_instruction(self, query: str, doc: str) -> str:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        
    def _process_inputs(self, pairs: List[str]):
        # Sử dụng return_attention_mask=False để tránh lệch kích thước attention_mask
        # khi ta chèn thêm prefix/suffix tokens. Hàm tokenizer.pad sẽ tự tính lại mask chuẩn xác.
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # Bọc input_ids vào cấu trúc đặc biệt của Reranker
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
            
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs
        
    def _compute_scores(self, inputs) -> List[float]:
        if torch is None or self.model is None:
            return [0.0] * len(inputs["input_ids"])
            
        with torch.no_grad():
            # Trích xuất logits của token cuối cùng đại diện cho xác suất yes/no
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            
            # Hợp nhất và chuẩn hóa phân phối qua log_softmax
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()  # Lấy xác suất của lớp "yes"
            return scores

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
            
        self.load_model()
        if self.model is None:
            # Fallback nếu không có GPU hoặc lỗi nạp mô hình: giữ nguyên và lấy top k
            print("[Cảnh báo] Cross-Encoder Reranker không khả dụng. Bỏ qua bước xếp hạng lại và cắt lấy top_k.")
            return documents[:self.top_k]
            
        try:
            pairs = [self._format_instruction(query, doc.page_content) for doc in documents]
            inputs = self._process_inputs(pairs)
            scores = self._compute_scores(inputs)
            
            # Gộp tài liệu và điểm số, sắp xếp giảm dần
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"[Reranker] Đã lọc xếp hạng lại {len(documents)} tài liệu thành {self.top_k} tài liệu tốt nhất.")
            # Cắt tỉa (Slicing) lấy top k
            return [doc for doc, score in doc_score_pairs[:self.top_k]]
        except Exception as e:
            print(f"[Reranker] Lỗi khi thực hiện xếp hạng lại: {str(e)}")
            return documents[:self.top_k]


class MMRReranker:
    """
    Tái xếp hạng dựa trên thuật toán Maximal Marginal Relevance (MMR) (Không cần GPU).
    Giúp đa dạng hóa thông tin retrieved context, loại bỏ các chunk bị trùng lặp nội dung.
    """
    def __init__(self, embedding_model: Any, top_k: int = 3, lambda_mult: float = 0.5):
        self.embeddings = embedding_model
        self.top_k = top_k
        self.lambda_mult = lambda_mult
        
    def _calculate_cosine_similarity(self, v1, v2) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
        
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents or len(documents) <= self.top_k:
            return documents
            
        try:
            # Lấy vector đại diện cho câu hỏi và các tài liệu
            query_embedding = np.array(self.embeddings.embed_query(query))
            doc_embeddings = [np.array(self.embeddings.embed_query(doc.page_content)) for doc in documents]
            
            selected_indices = []
            remaining_indices = list(range(len(documents)))
            
            # Chọn tài liệu đầu tiên giống query nhất
            similarities_to_query = [self._calculate_cosine_similarity(query_embedding, d_emb) for d_emb in doc_embeddings]
            best_idx = int(np.argmax(similarities_to_query))
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Vòng lặp tính toán MMR cho các tài liệu tiếp theo
            while len(selected_indices) < self.top_k and remaining_indices:
                best_mmr = -float("inf")
                best_candidate_idx = -1
                
                for candidate_idx in remaining_indices:
                    sim_to_query = similarities_to_query[candidate_idx]
                    
                    # Tìm độ tương đồng lớn nhất của ứng viên với các tài liệu đã chọn trước đó
                    max_sim_to_selected = max(
                        self._calculate_cosine_similarity(doc_embeddings[candidate_idx], doc_embeddings[selected_idx])
                        for selected_idx in selected_indices
                    )
                    
                    # Công thức MMR
                    mmr_score = self.lambda_mult * sim_to_query - (1 - self.lambda_mult) * max_sim_to_selected
                    
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_candidate_idx = candidate_idx
                        
                selected_indices.append(best_candidate_idx)
                remaining_indices.remove(best_candidate_idx)
                
            return [documents[idx] for idx in selected_indices]
            
        except Exception as e:
            print(f"[MMR Reranker] Lỗi khi thực hiện MMR: {str(e)}")
            return documents[:self.top_k]
