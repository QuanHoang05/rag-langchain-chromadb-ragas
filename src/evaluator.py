import os
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd

# Thư viện RAGAS để đánh giá tự động
try:
    from datasets import Dataset
    from ragas import evaluate
    # Metric KHONG cần ground_truth (dùng được khi không có nhãn chuẩn từ con người)
    from ragas.metrics import faithfulness, answer_relevancy
    # Metric CẦN có ground_truth chuẩn từ con người — KHONG được dùng output của LLM làm ground_truth!
    from ragas.metrics import context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False

class RAGLogger:
    """
    Ghi nhận dấu vết hoạt động (Observability Log) của hệ thống RAG nâng cao.
    Ghi dữ liệu hội thoại và ngữ cảnh vào file JSON cục bộ phục vụ bước đánh giá định lượng.
    """
    def __init__(self, log_dir: str = "./logs"):
        self.log_file = os.path.join(log_dir, "rag_traces.json")
        os.makedirs(log_dir, exist_ok=True)
        
    def log_trace(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str], 
        metadata_sources: List[Dict[str, Any]],
        execution_time: float,
        route: str,
        ground_truth: str = ""
    ):
        """
        Ghi lại một phiên truy vấn vào log file.
        """
        trace = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "sources": metadata_sources,
            "execution_time_seconds": execution_time,
            "route": route,
            "ground_truth": ground_truth  # Nếu có đáp án chuẩn phục vụ RAGAS
        }
        
        traces = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    traces = json.load(f)
            except Exception:
                traces = []
                
        traces.append(trace)
        
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(traces, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[Lỗi] Không thể ghi log trace RAG: {str(e)}")

    def get_all_traces(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

class RAGEvaluator:
    """
    Đánh giá chất lượng hệ thống RAG sử dụng framework RAGAS.
    Tự động phát hiện có ground_truth hay không để chọn bộ metric phù hợp.

    RAGAS có 2 nhóm metric hoàn toàn khác nhau:

    NHOM 1 — Reference-Free (Không cần nhãn chuẩn):
        - faithfulness:      LLM có trả lời trung thực với context không? (chống hallucination)
        - answer_relevancy:  Câu trả lời có đúng trịng tâm câu hỏi không?

    NHOM 2 — Reference-Based (Bắt buộc phải có đáp án chuẩn từ con người):
        - context_recall:    Context có chứa đủ thông tin mà đáp án chuẩn yêu cầu không?
        - context_precision: Trong context trị nhớ ra, bao nhiêu % thực sự cần thiết?

    [!CANH BAO] TUYET DOI KHONG dùng output của chính LLM làm ground_truth.
    Nếu làm vậy, LLM chỉ viết dựa trên context nó đã đọc, nên Context Recall
    sẽ luôn gần = 1.0 — "tự đá bóng, tự thổi còi" — kết quả hoàn toàn vô giá trị.
    """
    def __init__(self, output_dir: str = "./logs"):
        self.report_file = os.path.join(output_dir, "eval_report.json")
        os.makedirs(output_dir, exist_ok=True)
        
    def _has_ground_truth(self, test_dataset: List[Dict[str, Any]]) -> bool:
        """
        Kiểm tra xem tập dữ liệu có chứa ground_truth chuẩn từ con người không.
        Chỉ tính là có nếu ít nhất 50% mẫu dữ liệu có ground_truth không rỗng.
        """
        gt_count = sum(
            1 for item in test_dataset
            if item.get("ground_truth", "").strip()
        )
        return gt_count >= len(test_dataset) * 0.5
        
    def run_evaluation(
        self, 
        evaluator_llm: Any, 
        evaluator_embeddings: Any,
        test_dataset: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Chạy đánh giá Ragas trên một tập dữ liệu thử nghiệm.
        test_dataset là list dict, mỗi dict chứa:
          - 'question'    : Câu hỏi gốc
          - 'answer'      : Câu trả lời của hệ thống RAG cần đánh giá
          - 'contexts'    : List chuỗi ngữ cảnh đã truy xuất (để chấm Faithfulness)
          - 'ground_truth': [Tùy chọn] Đáp án chuẩn DO CON NGƯỜI viết — bắt buộc để
                            chạy Context Recall và Context Precision.
        """
        if not test_dataset:
            raise ValueError("Tập dữ liệu đánh giá trống. Cần truyền vào danh sách dữ liệu RAG.")
            
        if not _RAGAS_AVAILABLE:
            raise ImportError("Chưa cài đặt thư viện 'ragas'. Vui lòng chạy: pip install ragas")
            
        # Phát hiện tự động: có ground_truth chuẩn không?
        use_reference_metrics = self._has_ground_truth(test_dataset)
        
        if use_reference_metrics:
            # Chạy đầy đủ 4 metric khi có đáp án chuẩn từ con người
            metrics_to_run = [faithfulness, answer_relevancy, context_precision, context_recall]
            eval_mode = "Full (Reference-Based): Faithfulness + Answer Relevancy + Context Precision + Context Recall"
        else:
            # Chỉ chạy 2 metric không cần nhãn chuẩn khi không có ground_truth
            metrics_to_run = [faithfulness, answer_relevancy]
            eval_mode = "Reference-Free: Faithfulness + Answer Relevancy (không có ground_truth chuẩn)"
            
        print(f"Bắt đầu đánh giá RAGAS trên {len(test_dataset)} mẫu dữ liệu...")
        print(f"Chế độ đánh giá: {eval_mode}")
        
        # Chuẩn bị dữ liệu đầu vào cho RAGAS
        data_dict = {
            "question":    [item["question"] for item in test_dataset],
            "answer":      [item["answer"] for item in test_dataset],
            "contexts":    [item["contexts"] for item in test_dataset],
            "ground_truth": [item.get("ground_truth", "") for item in test_dataset]
        }
        
        dataset = Dataset.from_dict(data_dict)
        
        # Bọc LLM và Embeddings của LangChain để chạy qua RAGAS
        wrapped_llm = LangchainLLMWrapper(evaluator_llm)
        wrapped_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)
        
        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_run,
                llm=wrapped_llm,
                embeddings=wrapped_embeddings,
                raise_exceptions=False
            )
            
            scores = dict(results)
            print("Đánh giá RAGAS hoàn tất! Scores:", scores)
            
            report = {
                "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "samples_evaluated": len(test_dataset),
                "eval_mode": eval_mode,
                "has_ground_truth": use_reference_metrics,
                "scores": scores,
                "status": "success",
                "note": (
                    None if use_reference_metrics 
                    else (
                        "Context Recall và Context Precision không được chạy vì thiếu ground_truth "
                        "chuẩn từ con người. Cung cấp 'ground_truth' cho từng mẫu để chạy được."
                    )
                )
            }
            
            with open(self.report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
                
            return report
        except Exception as e:
            print(f"[Lỗi] Thất bại khi chạy Ragas Evaluation: {str(e)}")
            return {"status": "failed", "error": str(e)}
            
    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.report_file):
            return None
        try:
            with open(self.report_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
