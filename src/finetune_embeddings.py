import os
import json
import time
from typing import List, Tuple, Any, Dict
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

# Thư viện huấn luyện Embedding
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
except ImportError:
    pass

class EmbeddingFinetuner:
    """
    Quy trình Huấn luyện tinh chỉnh Embedding (Embedding Fine-Tuning Pipeline):
    1. Đọc các chunks tài liệu nội bộ.
    2. Dùng LLM sinh câu hỏi tương ứng với từng chunk để làm dữ liệu giám sát tích cực (Question, Chunks).
    3. Huấn luyện mô hình Embedding (SentenceTransformer) sử dụng hàm mất mát MultipleNegativesRankingLoss.
    4. Lưu mô hình đã tinh chỉnh vào thư mục cục bộ để cắm lại vào RAG pipeline.
    """
    def __init__(self, llm: BaseChatModel, base_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.llm = llm
        self.base_model_name = base_model_name
        self.output_dir = "./models/fine_tuned_embeddings"
        
        # Prompt tạo câu hỏi giả định ngắn gọn
        self.question_generator_prompt = """Bạn là một chuyên gia khảo thí. Hãy đọc đoạn văn dưới đây và viết đúng 1 câu hỏi duy nhất mà đoạn văn này có thể trả lời trực tiếp một cách đầy đủ.
Yêu cầu câu hỏi tự nhiên bằng tiếng Việt, đi thẳng vào ý chính, không chứa thông tin thừa.
Không viết thêm giải thích nào khác.

Đoạn văn:
"{context}"

Câu hỏi tiếng Việt:"""

    def generate_synthetic_dataset(self, chunks: List[Document], max_samples: int = 50) -> List[Tuple[str, str]]:
        """
        Dùng LLM sinh tập dữ liệu câu hỏi - ngữ cảnh tương ứng.
        Giới hạn số lượng mẫu để tiết kiệm chi phí gọi API và thời gian chạy.
        """
        print(f"Bắt đầu tạo dữ liệu huấn luyện tổng hợp (Synthetic Dataset) từ {min(len(chunks), max_samples)} chunks...")
        training_pairs = []
        
        # Chọn mẫu ngẫu nhiên hoặc lấy các chunks đầu tiên
        sampled_chunks = chunks[:max_samples]
        
        for idx, chunk in enumerate(tqdm(sampled_chunks, desc="LLM Generating Questions")):
            context = chunk.page_content
            # Giới hạn nội dung gửi đi để tiết kiệm input tokens
            prompt = self.question_generator_prompt.format(context=context[:800])
            
            try:
                response = self.llm.invoke(prompt)
                question = response.content.strip()
                # Làm sạch phản hồi
                question = question.replace('"', '').replace("'", "").strip()
                
                if len(question) > 10:
                    training_pairs.append((question, context))
                
                # Tránh lỗi rate limit của API free
                time.sleep(0.5)
            except Exception as e:
                print(f"\n[Lỗi] Lỗi sinh câu hỏi tại chunk {idx}: {str(e)}")
                continue
                
        print(f"Đã tạo thành công {len(training_pairs)} cặp (Câu hỏi, Ngữ cảnh) làm tập dữ liệu huấn luyện.")
        return training_pairs

    def run_finetuning(self, training_pairs: List[Tuple[str, str]], epochs: int = 1, batch_size: int = 8) -> str:
        """
        Huấn luyện mô hình SentenceTransformer cục bộ.
        """
        if not training_pairs:
            raise ValueError("Tập dữ liệu huấn luyện trống. Không thể chạy fine-tuning.")
            
        if 'SentenceTransformer' not in globals():
            raise ImportError("Chưa cài đặt thư viện 'sentence-transformers'. Vui lòng chạy pip install trước.")
            
        print(f"Khởi tạo mô hình nền tảng: {self.base_model_name}...")
        model = SentenceTransformer(self.base_model_name)
        
        # Đóng gói dữ liệu thành định dạng huấn luyện của SentenceTransformer
        train_examples = []
        for question, context in training_pairs:
            train_examples.append(InputExample(texts=[question, context]))
            
        # Nạp dữ liệu qua DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Sử dụng MultipleNegativesRankingLoss (Hàm mất mát hiệu quả nhất cho RAG không cần mẫu tiêu cực cứng)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        print(f"Bắt đầu huấn luyện tinh chỉnh mô hình (Epochs={epochs}, Batch Size={batch_size})...")
        start_time = time.time()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Thực thi huấn luyện
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            show_progress_bar=True
        )
        
        # Lưu mô hình đã tinh chỉnh thành công
        model.save(self.output_dir)
        elapsed_time = time.time() - start_time
        print(f"Huấn luyện hoàn tất sau {elapsed_time:.2f} giây! Mô hình đã được lưu tại: {self.output_dir}")
        
        # Lưu báo cáo huấn luyện
        report = {
            "base_model": self.base_model_name,
            "training_samples": len(training_pairs),
            "epochs": epochs,
            "training_time_seconds": elapsed_time,
            "status": "success"
        }
        with open(os.path.join(self.output_dir, "training_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        return self.output_dir

    def evaluate_embeddings(
        self, 
        model_or_path: Any, 
        eval_pairs: List[Tuple[str, str]], 
        k: int = 5
    ) -> Dict[str, float]:
        """
        Đánh giá chất lượng mô hình Embedding dựa trên các chỉ số tìm kiếm:
        - Hit Rate@K: Tỉ lệ tìm kiếm thấy ngữ cảnh chính xác nằm trong Top K kết quả.
        - MRR@K (Mean Reciprocal Rank): Điểm thứ hạng nghịch đảo trung bình của ngữ cảnh chính xác.
        """
        if not eval_pairs:
            return {"hit_rate": 0.0, "mrr": 0.0}
            
        # Nạp mô hình SentenceTransformer nếu truyền vào đường dẫn string
        if isinstance(model_or_path, str):
            if not os.path.exists(model_or_path):
                print(f"[Đánh giá] Đường dẫn mô hình không tồn tại: {model_or_path}")
                return {"hit_rate": 0.0, "mrr": 0.0}
            model = SentenceTransformer(model_or_path)
        else:
            model = model_or_path
            
        try:
            questions = [pair[0] for pair in eval_pairs]
            # Tạo tập corpus các văn bản độc nhất để làm không gian tìm kiếm
            corpus = list(set([pair[1] for pair in eval_pairs]))
            
            # Mã hóa vector
            q_embs = model.encode(questions, show_progress_bar=False)
            c_embs = model.encode(corpus, show_progress_bar=False)
            
            import numpy as np
            
            # Chuẩn hóa vector L2 norm để tính Cosine Similarity bằng Dot Product
            q_norms = np.linalg.norm(q_embs, axis=1, keepdims=True)
            c_norms = np.linalg.norm(c_embs, axis=1, keepdims=True)
            
            # Tránh chia cho 0
            q_norms[q_norms == 0] = 1e-10
            c_norms[c_norms == 0] = 1e-10
            
            q_embs = q_embs / q_norms
            c_embs = c_embs / c_norms
            
            # Matrix tương đồng shape (num_questions, num_corpus)
            similarity_matrix = np.dot(q_embs, c_embs.T)
            
            hits = 0
            mrr_sum = 0.0
            
            for i, (q, correct_ctx) in enumerate(eval_pairs):
                correct_idx = corpus.index(correct_ctx)
                
                # Sắp xếp chỉ số corpus theo điểm tương đồng giảm dần
                scores = similarity_matrix[i]
                sorted_indices = np.argsort(scores)[::-1]
                
                # Tìm vị trí xếp hạng của correct_idx (0-indexed -> 1-indexed)
                rank = np.where(sorted_indices == correct_idx)[0][0] + 1
                
                if rank <= k:
                    hits += 1
                    mrr_sum += 1.0 / rank
                    
            hit_rate = hits / len(eval_pairs)
            mrr = mrr_sum / len(eval_pairs)
            
            return {
                "hit_rate": round(hit_rate, 4),
                "mrr": round(mrr, 4)
            }
        except Exception as e:
            print(f"[Đánh giá] Lỗi tính toán chỉ số embedding: {str(e)}")
            return {"hit_rate": 0.0, "mrr": 0.0}

