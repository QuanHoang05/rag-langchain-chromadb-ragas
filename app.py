import os
import sys
import time
import shutil
import nest_asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from typing import List, Dict, Any

# Đảm bảo import src.* hoạt động đúng dù chạy từ bất kỳ thư mục nào (Docker, Colab, local)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Kích hoạt nest_asyncio để cho phép chạy event loop lặp trong môi trường Jupyter/Notebook
nest_asyncio.apply()

import asyncio
# Cấu hình bản vá động tương thích ngược cho asyncio.run với loop_factory (khắc phục xung đột uvicorn trên Python 3.11+)
try:
    _orig_run = asyncio.run
    def _safe_run(main, *, debug=False, loop_factory=None):
        try:
            if loop_factory is not None:
                loop = loop_factory()
                asyncio.set_event_loop(loop)
            # Chuyển tiếp an toàn, loại bỏ loop_factory để nest_asyncio không gặp lỗi signature
            return _orig_run(main, debug=debug)
        except RuntimeError as e:
            if "already running" in str(e):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(main)
            raise e
    asyncio.run = _safe_run
except Exception as e:
    print(f"Cảnh báo: Không thể vá lỗi asyncio.run: {e}")

# Import các thành phần nội bộ
from src.loaders import SimpleLoader
from src.chunkers import SemanticChunker, FixedSizeChunker
from src.retrievers import VectorStoreManager
from src.chain import AdvancedRAGChain
from src.finetune_embeddings import EmbeddingFinetuner
from src.evaluator import RAGEvaluator, RAGLogger

# Các thư viện LangChain và mô hình
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Thiết lập mặc định - ROOT_DIR tự động phát hiện theo vị trí của file app.py
# Hoạt động đúng trên cả Windows cuc̣c bọ, Docker (mount taại /app) và Google Colab
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(ROOT_DIR, "Ducument")
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Tạo các thư mục nếu chưa tồn tại
for d in [DOC_DIR, CHROMA_DIR, LOG_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Khởi tạo mô hình Embedding mặc định (SBERT đa ngôn ngữ)
# Sẽ tự động dùng mô hình fine-tuned nếu đã được huấn luyện thành công
def get_active_embeddings(use_finetuned: bool = True):
    finetuned_path = os.path.join(MODELS_DIR, "fine_tuned_embeddings")
    if use_finetuned and os.path.exists(os.path.join(finetuned_path, "model.safetensors")) or os.path.exists(os.path.join(finetuned_path, "pytorch_model.bin")):
        print(f"[Embedding] Đang sử dụng mô hình Fine-tuned từ: {finetuned_path}")
        return HuggingFaceEmbeddings(model_name=finetuned_path)
    
    # Mặc định sử dụng mô hình đa ngôn ngữ gọn nhẹ từ Hugging Face
    print("[Embedding] Đang sử dụng mô hình mặc định: paraphrase-multilingual-MiniLM-L12-v2")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Quản lý Vector DB toàn cục
embeddings = get_active_embeddings()
vector_manager = VectorStoreManager(persist_directory=CHROMA_DIR, embedding_model=embeddings)

# Tự động nạp tài liệu mặc định nếu database trống
try:
    vector_manager.init_db()
    # Kiểm tra xem collection có dữ liệu chưa
    db_docs = vector_manager.db.get()["documents"]
    if not db_docs:
        print("[Ingestion] Database trống. Khởi động nạp tài liệu tự động...")
        loader = SimpleLoader()
        # Đọc toàn bộ file trong Ducument/ (bao gồm AI, LuatDatDai, LuatLaoDong)
        raw_docs = loader.load_all_folders(DOC_DIR)
        if raw_docs:
            # Sử dụng Fixed Chunker tạm thời để nạp nhanh ban đầu
            chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=100)
            chunks = chunker.split(raw_docs)
            vector_manager.init_db(chunks)
            print(f"[Ingestion] Đã nạp thành công {len(chunks)} chunks tài liệu ban đầu!")
except Exception as e:
    print(f"[Cảnh báo] Lỗi khởi tạo database ban đầu: {str(e)}")

# Hàm chính xử lý câu hỏi
def rag_qa_interface(
    query: str,
    llm_provider: str,
    api_key: str,
    chunker_type: str,
    search_mode: str,
    fusion_mode: str,
    query_transform: str,
    reranker_type: str,
    use_router: bool,
    force_internet: bool,
    selected_tags: List[str]
):
    if not query.strip():
        return "Vui lòng nhập câu hỏi.", "", "Không có trích dẫn."

    # Khởi tạo mô hình LLM dựa trên nhà cung cấp được chọn
    try:
        if llm_provider == "Gemini API":
            if not api_key:
                # Tìm trong biến môi trường nếu không nhập key trực tiếp
                api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return "Lỗi: Vui lòng nhập API Key cho Gemini.", "", "Không có trích dẫn."
            os.environ["GEMINI_API_KEY"] = api_key
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_version="v1", temperature=0.2)
            
        elif llm_provider == "OpenAI API":
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return "Lỗi: Vui lòng nhập API Key cho OpenAI.", "", "Không có trích dẫn."
            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            
        else:
            return "Lỗi: Nhà cung cấp mô hình chưa được hỗ trợ cục bộ trong phiên bản này.", "", "Không có trích dẫn."
    except Exception as e:
        return f"Lỗi khởi tạo LLM: {str(e)}", "", "Không có trích dẫn."

    # Cấu hình tham số RAG
    config = {
        "search_mode": search_mode,
        "fusion_mode": fusion_mode,
        "query_transform": query_transform,
        "reranker_type": reranker_type,
        "use_router": use_router,
        "force_internet": force_internet,
        "doc_tags": selected_tags
    }

    try:
        # Khởi tạo chuỗi xích RAG nâng cao
        chain = AdvancedRAGChain(
            llm=llm,
            embeddings=embeddings,
            vector_manager=vector_manager,
            log_dir=LOG_DIR
        )
        
        # Chạy suy luận
        result = chain.run(query, config)
        
        answer = result["answer"]
        execution_time = result["execution_time_seconds"]
        logs = result["logs"]
        citations = result["citations"]
        
        # Format nhật ký xử lý chi tiết (Observability Logs)
        log_html = f"<b>Kênh định tuyến (Route Decision):</b> {logs.get('route_decision', 'N/A')}<br>"
        log_html += f"<b>Nguồn tài liệu:</b> {logs.get('retrieval_source', 'N/A')}<br>"
        log_html += f"<b>Tổng số chunks truy xuất ban đầu:</b> {logs.get('raw_retrieved_count', 0)} chunks<br>"
        log_html += f"<b>Số chunks sau khi lọc & rerank:</b> {logs.get('reranked_count', 0)} chunks<br>"
        log_html += f"<b>Thời gian xử lý:</b> {execution_time:.2f} giây<br>"
        
        if "hyde_document" in logs:
            log_html += f"<br><b>Tài liệu giả định HyDE:</b><br><i>{logs['hyde_document']}</i><br>"
        if "decomposed_queries" in logs:
            log_html += f"<br><b>Các câu hỏi phụ phân rã:</b><br><li>" + "</li><li>".join(logs['decomposed_queries']) + "</li><br>"

        # Format trích dẫn nguồn (Citations Card Panel)
        citations_html = ""
        if not citations:
            citations_html = "<div style='color: gray;'>Không có tài liệu trích dẫn phù hợp.</div>"
        else:
            for idx, c in enumerate(citations):
                citations_html += f"""
                <div style='border: 1px solid #444; border-radius: 8px; padding: 12px; margin-bottom: 10px; background-color: #1e1e1e;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                        <span style='color: #4CAF50; font-weight: bold;'>[Nguồn {idx+1}] {c['source']}</span>
                        <span style='background-color: #333; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 11px;'>Tag: {c['tag']} | Trang {c['page']}</span>
                    </div>
                    <div style='font-style: italic; color: #ccc; font-size: 13px;'>"{c['snippet']}"</div>
                </div>
                """

        return answer, log_html, citations_html
        
    except Exception as e:
        return f"Lỗi thực thi RAG Chain: {str(e)}", f"Lỗi: {str(e)}", "Lỗi trích dẫn."

# Quản lý danh sách tài liệu hiện có trên UI
def list_indexed_documents():
    try:
        data = vector_manager.db.get()
        metadatas = data["metadatas"]
        if not metadatas:
            return pd.DataFrame(columns=["Tên File", "Nhãn Tag", "Số Chunks"])
            
        df_list = []
        file_counts = {}
        file_tags = {}
        
        for m in metadatas:
            file_name = m.get("source_name", "Không rõ")
            tag = m.get("doc_tag", "Chưa phân loại")
            file_counts[file_name] = file_counts.get(file_name, 0) + 1
            file_tags[file_name] = tag
            
        for file_name, count in file_counts.items():
            df_list.append({
                "Tên File": file_name,
                "Nhãn Tag": file_tags[file_name],
                "Số Chunks": count
            })
            
        return pd.DataFrame(df_list)
    except Exception as e:
        print(f"Lỗi đọc tài liệu: {str(e)}")
        return pd.DataFrame(columns=["Tên File", "Nhãn Tag", "Số Chunks"])

# Hàm tải file mới lên VectorDB
def upload_file_interface(files, custom_tag):
    if not files:
        return "Vui lòng chọn file.", list_indexed_documents()
        
    if not custom_tag.strip():
        custom_tag = "Upload"
        
    loader = SimpleLoader()
    # Sử dụng Semantic Chunker để chia nhỏ tài liệu mới tải lên
    chunker = SemanticChunker(
        embedding_model=embeddings,
        breakpoint_threshold=0.5,
        max_chunk_tokens=350
    )
    
    total_chunks = 0
    try:
        for file in files:
            raw_docs = loader.load_file(file.name, tag=custom_tag)
            chunks = chunker.split(raw_docs)
            if chunks:
                vector_manager.db.add_documents(chunks)
                total_chunks += len(chunks)
                
        return f"Đã nạp thành công {len(files)} files với tổng cộng {total_chunks} chunks mới!", list_indexed_documents()
    except Exception as e:
        return f"Lỗi khi nạp file: {str(e)}", list_indexed_documents()

# Hàm chạy fine-tuning Embedding
def run_finetuning_ui(llm_provider, api_key, sample_size, epochs, batch_size):
    global embeddings, vector_manager
    # Khởi tạo LLM cho sinh câu hỏi giả định
    try:
        if llm_provider == "Gemini API":
            if not api_key:
                api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return "Thất bại. Vui lòng điền Gemini API Key để sinh tập dữ liệu."
            os.environ["GEMINI_API_KEY"] = api_key
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_version="v1")
        elif llm_provider == "OpenAI API":
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return "Thất bại. Vui lòng điền OpenAI API Key để sinh tập dữ liệu."
            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(model="gpt-4o-mini")
        else:
            return "Thất bại. Chưa cấu hình API LLM cho fine-tuning."
    except Exception as e:
        return f"Lỗi khởi tạo LLM: {str(e)}"

    # Lấy tài liệu hiện có làm ngữ cảnh huấn luyện
    try:
        data = vector_manager.db.get()
        raw_texts = data["documents"]
        metadatas = data["metadatas"]
        if not raw_texts:
            return "Thất bại. Cơ sở dữ liệu trống, vui lòng nạp tài liệu trước."
            
        chunks = [Document(page_content=text, metadata=meta) for text, meta in zip(raw_texts, metadatas)]
    except Exception as e:
        return f"Lỗi lấy dữ liệu: {str(e)}"

    # Chạy quy trình tinh chỉnh
    try:
        finetuner = EmbeddingFinetuner(llm=llm)
        training_pairs = finetuner.generate_synthetic_dataset(chunks, max_samples=int(sample_size))
        
        if not training_pairs:
            return "Không thể sinh cặp dữ liệu huấn luyện nào. Vui lòng kiểm tra lại kết nối API."
            
        # Phân chia dữ liệu Train / Test (80% / 20%) để đánh giá khách quan
        import random
        random.seed(42)
        random.shuffle(training_pairs)
        
        split_idx = int(len(training_pairs) * 0.8)
        if split_idx == len(training_pairs) and len(training_pairs) >= 2:
            split_idx = len(training_pairs) - 1
        elif split_idx == 0 and len(training_pairs) >= 2:
            split_idx = 1
            
        train_pairs = training_pairs[:split_idx]
        test_pairs = training_pairs[split_idx:] if split_idx < len(training_pairs) else training_pairs
        
        # 1. Đánh giá mô hình nền trước khi huấn luyện
        from sentence_transformers import SentenceTransformer
        try:
            base_model = SentenceTransformer(finetuner.base_model_name)
            pre_eval = finetuner.evaluate_embeddings(base_model, test_pairs, k=5)
        except Exception as e:
            pre_eval = {"hit_rate": 0.0, "mrr": 0.0}
            print(f"[Cảnh báo] Không thể đánh giá trước huấn luyện: {str(e)}")
            
        # 2. Huấn luyện tinh chỉnh mô hình
        model_path = finetuner.run_finetuning(train_pairs, epochs=int(epochs), batch_size=int(batch_size))
        
        # 3. Đánh giá mô hình sau khi huấn luyện
        try:
            post_eval = finetuner.evaluate_embeddings(model_path, test_pairs, k=5)
        except Exception as e:
            post_eval = {"hit_rate": 0.0, "mrr": 0.0}
            print(f"[Cảnh báo] Không thể đánh giá sau huấn luyện: {str(e)}")
            
        # Nạp lại mô hình embedding mới vào bộ nhớ toàn cục
        embeddings = HuggingFaceEmbeddings(model_name=model_path)
        vector_manager = VectorStoreManager(persist_directory=CHROMA_DIR, embedding_model=embeddings)
        vector_manager.init_db()
        
        # Tạo báo cáo kết quả so sánh định lượng trực quan
        report_msg = f"🎉 Quá trình tinh chỉnh (Fine-tuning) hoàn tất thành công!\n"
        report_msg += f"📁 Mô hình lưu tại: {model_path}\n\n"
        report_msg += f"📊 KẾT QUẢ ĐÁNH GIÁ ĐẦU RA EMBEDDING (Trên {len(test_pairs)} mẫu kiểm thử độc lập):\n"
        report_msg += f"- TRƯỚC KHI TINH CHỈNH (Mô hình nền):\n"
        report_msg += f"  • Tỉ lệ tìm thấy đúng (Hit Rate@5): {pre_eval['hit_rate']*100:.2f}%\n"
        report_msg += f"  • Thứ hạng trung bình (MRR@5): {pre_eval['mrr']*100:.2f}%\n"
        report_msg += f"- SAU KHI TINH CHỈNH (Mô hình đã Fine-tune):\n"
        report_msg += f"  • Tỉ lệ tìm thấy đúng (Hit Rate@5): {post_eval['hit_rate']*100:.2f}%\n"
        report_msg += f"  • Thứ hạng trung bình (MRR@5): {post_eval['mrr']*100:.2f}%\n\n"
        
        hit_diff = post_eval['hit_rate'] - pre_eval['hit_rate']
        mrr_diff = post_eval['mrr'] - pre_eval['mrr']
        report_msg += f"📈 Hiệu năng thay đổi: Hit Rate: {hit_diff*100:+.2f}%, MRR: {mrr_diff*100:+.2f}%\n"
        report_msg += f"Hệ thống RAG đã được cập nhật tự động để sử dụng mô hình tinh chỉnh mới!"
        
        return report_msg
    except Exception as e:
        return f"Lỗi trong quá trình huấn luyện: {str(e)}"

# Chạy đánh giá Ragas
def run_evaluation_ui(llm_provider, api_key):
    try:
        if llm_provider == "Gemini API":
            if not api_key:
                api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return "Vui lòng nhập Gemini API Key để làm giám khảo Ragas.", None
            os.environ["GEMINI_API_KEY"] = api_key
            eval_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_version="v1", temperature=0.0)
        elif llm_provider == "OpenAI API":
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return "Vui lòng nhập OpenAI API Key để làm giám khảo Ragas.", None
            os.environ["OPENAI_API_KEY"] = api_key
            eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        else:
            return "Ragas chưa hỗ trợ giám khảo offline trong UI này.", None
    except Exception as e:
        return f"Lỗi khởi tạo LLM chấm điểm: {str(e)}", None

    # Lấy các trace logs hoạt động
    logger = RAGLogger(LOG_DIR)
    traces = logger.get_all_traces()
    
    if not traces:
        return "Không tìm thấy logs hội thoại nào. Hãy hỏi hệ thống RAG vài câu hỏi ở Tab 1 để ghi nhận logs trước khi đánh giá.", None
        
    # Tạo tập test tối đa 10 mẫu để tránh tốn API
    eval_dataset = []
    for t in traces[-10:]:
        # Giả định câu trả lời chuẩn (ground_truth) bằng câu trả lời hiện tại nếu trống
        eval_dataset.append({
            "question": t["question"],
            "answer": t["answer"],
            "contexts": t["contexts"],
            "ground_truth": t.get("ground_truth", t["answer"])
        })
        
    try:
        evaluator = RAGEvaluator(LOG_DIR)
        report = evaluator.run_evaluation(
            evaluator_llm=eval_llm,
            evaluator_embeddings=embeddings,
            test_dataset=eval_dataset
        )
        
        if report.get("status") == "success":
            scores = report["scores"]
            
            # Lưu biểu đồ hình cột
            plt.figure(figsize=(8, 5))
            sns.set_theme(style="whitegrid")
            
            metrics = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
            values = [
                scores.get("faithfulness", 0),
                scores.get("answer_relevancy", 0),
                scores.get("context_precision", 0),
                scores.get("context_recall", 0)
            ]
            
            ax = sns.barplot(x=metrics, y=values, palette="viridis")
            plt.title("Đánh giá RAGAS Benchmark", fontsize=14, fontweight="bold")
            plt.ylabel("Điểm số (0 - 1)")
            plt.ylim(0, 1)
            
            # Thêm nhãn số trên các cột
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                            ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                            textcoords='offset points')
                            
            chart_path = os.path.join(LOG_DIR, "ragas_chart.png")
            plt.savefig(chart_path, bbox_inches="tight")
            plt.close()
            
            # Format markdown bảng điểm
            md_table = f"""
            ### Kết quả RAGAS Benchmark (Tính trên {len(eval_dataset)} câu hỏi gần nhất)
            
            | Metric chỉ số | Điểm số (0.0 - 1.0) | Giải nghĩa |
            | :--- | :---: | :--- |
            | **Faithfulness (Độ trung thực)** | `{scores.get('faithfulness', 0):.4f}` | Đo lượng thông tin bịa đặt (hallucination) |
            | **Answer Relevancy (Độ liên quan)** | `{scores.get('answer_relevancy', 0):.4f}` | Đo việc trả lời đúng trọng tâm câu hỏi |
            | **Context Precision (Độ chính xác)** | `{scores.get('context_precision', 0):.4f}` | Đo chất lượng sắp xếp vị trí chunks tìm được |
            | **Context Recall (Độ bao phủ)** | `{scores.get('context_recall', 0):.4f}` | Đo mức độ bao phủ đầy đủ ý của dữ liệu nguồn |
            """
            
            return md_table, chart_path
        else:
            return f"Lỗi Ragas: {report.get('error')}", None
            
    except Exception as e:
        return f"Lỗi chạy đánh giá: {str(e)}", None

# Reset cơ sở dữ liệu về mặc định ban đầu
def reset_database_ui():
    try:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        # Nạp lại mặc định
        global vector_manager
        vector_manager = VectorStoreManager(persist_directory=CHROMA_DIR, embedding_model=embeddings)
        vector_manager.init_db()
        return "Đã xóa toàn bộ Vector Database. Vui lòng nạp lại tài liệu mới.", list_indexed_documents()
    except Exception as e:
        return f"Lỗi reset DB: {str(e)}", list_indexed_documents()


# XÂY DỰNG GIAO DIỆN GRADIO (PREMIUM DARK MODE THEME)
custom_css = """
body { background-color: #121212; color: #e0e0e0; font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
.tabs { border-bottom: 2px solid #333 !important; }
.tab-navbutton { font-size: 16px !important; font-weight: 600 !important; color: #aaa !important; }
.tab-navbutton-active { color: #4CAF50 !important; border-bottom: 3px solid #4CAF50 !important; }
.input-box textarea { background-color: #1e1e1e !important; color: #fff !important; border: 1px solid #444 !important; }
.btn-primary { background-color: #4CAF50 !important; border: none !important; color: #white !important; }
.btn-primary:hover { background-color: #45a049 !important; }
"""

with gr.Blocks(title="Advanced Vietnamese RAG System") as demo:
    gr.Markdown("""
    # 🧠 Hệ thống Hỏi Đáp Tài Liệu Tiếng Việt Nâng Cao (Advanced RAG)
    Ứng dụng được thiết kế tối ưu hóa **Token 4 Giai đoạn** để tiết kiệm chi phí API, hỗ trợ Fine-tuning Embedding và kiểm thử định lượng bằng RAGAS.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Cấu hình API và Nhà cung cấp LLM
            llm_provider = gr.Dropdown(
                choices=["Gemini API", "OpenAI API"],
                value="Gemini API",
                label="🔑 Nhà cung cấp Mô hình (LLM Provider)"
            )
            api_key = gr.Textbox(
                type="password",
                placeholder="Nhập API Key ở đây (hoặc cấu hình ENV)",
                label="API Key"
            )
            
        with gr.Column(scale=2):
            gr.Markdown("""
            > [!TIP]
            > - **Gemini API** sử dụng mô hình `gemini-1.5-flash` hoạt động rất tốt với ngôn ngữ Tiếng Việt và có chi phí rất thấp.
            > - Bạn có thể bỏ qua ô điền API Key nếu đã thiết lập biến môi trường `GEMINI_API_KEY` hoặc `OPENAI_API_KEY` trong file cấu hình Docker.
            """)

    with gr.Tabs():
        # TAB 1: HỎI ĐÁP NÂNG CAO
        with gr.Tab("💬 Hỏi Đáp Nâng Cao"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Chiến lược RAG Nâng Cao")
                    
                    use_router = gr.Checkbox(value=True, label="Kích hoạt Bộ định tuyến (Query Router)")
                    force_internet = gr.Checkbox(value=False, label="Ép buộc Tra cứu Internet (Web Search)")
                    
                    chunker_type = gr.Radio(
                        choices=["Fixed Size Chunker", "Semantic Chunker"],
                        value="Semantic Chunker",
                        label="1. Bộ chia nhỏ văn bản (Chunker)"
                    )
                    
                    search_mode = gr.Radio(
                        choices=["dense", "sparse", "hybrid"],
                        value="hybrid",
                        label="2. Phương thức tìm kiếm (Retrieval Mode)"
                    )
                    
                    fusion_mode = gr.Radio(
                        choices=["rrf", "interleave"],
                        value="rrf",
                        label="3. Thuật toán hợp nhất (Fusion Mode)"
                    )
                    
                    query_transform = gr.Radio(
                        choices=["none", "hyde", "decomposition"],
                        value="none",
                        label="4. Biến đổi truy vấn (Query Transform)"
                    )
                    
                    reranker_type = gr.Radio(
                        choices=["none", "cross_encoder", "mmr"],
                        value="none",
                        label="5. Tái xếp hạng (Reranker)"
                    )
                    
                    # Trích xuất danh sách tags hiện có để lọc
                    tag_choices = ["AI", "LuatDatDai", "LuatLaoDong", "Upload"]
                    selected_tags = gr.CheckboxGroup(
                        choices=tag_choices,
                        value=[],
                        label="🏷️ Lọc theo nhãn tài liệu (Pre-filtering)"
                    )

                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        lines=3,
                        placeholder="Ví dụ: Có bao nhiêu module trong khoá học AIO2025? Hoặc quy định làm thêm giờ luật lao động thế nào?",
                        label="Nhập câu hỏi của bạn:"
                    )
                    submit_btn = gr.Button("Gửi câu hỏi", variant="primary")
                    
                    gr.Markdown("### 📑 Câu trả lời từ AI:")
                    answer_output = gr.Markdown(value="Câu trả lời sẽ hiển thị ở đây...")
                    
                    with gr.Accordion("🔍 Nhật ký xử lý & Ý định RAG (Observability Logs)", open=False):
                        log_output = gr.HTML(value="Nhật ký xử lý chi tiết...")
                        
                    gr.Markdown("### 🔗 Tài liệu trích dẫn (Citations):")
                    citations_output = gr.HTML(value="Các nguồn trích dẫn sẽ hiển thị dưới dạng card ở đây...")

            # Kết nối sự kiện hỏi đáp
            submit_btn.click(
                fn=rag_qa_interface,
                inputs=[
                    query_input, llm_provider, api_key, chunker_type, 
                    search_mode, fusion_mode, query_transform, reranker_type,
                    use_router, force_internet, selected_tags
                ],
                outputs=[answer_output, log_output, citations_output]
            )

        # TAB 2: QUẢN LÝ TÀI LIỆU
        with gr.Tab("📁 Quản Lý Tài Liệu"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Tải lên tài liệu mới")
                    uploaded_files = gr.File(
                        file_count="multiple",
                        file_types=[".pdf", ".docx"],
                        label="Chọn file PDF hoặc DOCX"
                    )
                    custom_tag = gr.Textbox(value="Upload", label="Nhãn Tag gán cho tài liệu (ví dụ: LuatDatDai, AI,...)")
                    upload_btn = gr.Button("Thêm tài liệu vào VectorDB", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### ⚠️ Khu vực nguy hiểm")
                    reset_db_btn = gr.Button("Xóa toàn bộ CSDL (Reset VectorDB)", variant="stop")

                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Các tài liệu đang được lập chỉ mục")
                    doc_table = gr.Dataframe(value=list_indexed_documents(), interactive=False)
                    refresh_btn = gr.Button("Làm mới danh sách")

            # Kết nối các sự kiện
            upload_btn.click(
                fn=upload_file_interface,
                inputs=[uploaded_files, custom_tag],
                outputs=[gr.Textbox(label="Thông báo"), doc_table]
            )
            refresh_btn.click(fn=list_indexed_documents, outputs=[doc_table])
            reset_db_btn.click(fn=reset_database_ui, outputs=[gr.Textbox(label="Thông báo"), doc_table])

        # TAB 3: FINE-TUNING & ĐÁNH GIÁ (RAGAS)
        with gr.Tab("📊 Huấn luyện & Đánh giá (Benchmarks)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Fine-tune Embedding của riêng bạn")
                    gr.Markdown("Huấn luyện mô hình Embedding (SentenceTransformer) để tối ưu độ tương đồng ngữ nghĩa riêng cho tập dữ liệu của bạn.")
                    
                    sample_size = gr.Slider(minimum=10, maximum=150, value=30, step=5, label="Số lượng chunk sinh câu hỏi (Sample Size)")
                    epochs = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Số chu kỳ huấn luyện (Epochs)")
                    batch_size = gr.Slider(minimum=4, maximum=32, value=8, step=4, label="Kích thước lô huấn luyện (Batch Size)")
                    
                    run_ft_btn = gr.Button("Bắt đầu sinh dữ liệu & Huấn luyện", variant="primary")
                    ft_status = gr.Textbox(value="Chưa khởi chạy...", label="Trạng thái huấn luyện")

                with gr.Column(scale=1):
                    gr.Markdown("### 📉 Chạy Đánh giá RAGAS (Benchmarks)")
                    gr.Markdown("Hệ thống sẽ lấy tự động tối đa 10 câu hỏi gần nhất từ trace logs hoạt động ở Tab 1 để tự động chấm điểm độ chính xác bằng mô hình giám khảo.")
                    
                    run_eval_btn = gr.Button("Chạy chấm điểm RAGAS", variant="primary")
                    eval_results_table = gr.Markdown(value="Bảng điểm sẽ hiển thị ở đây...")
                    eval_chart = gr.Image(label="Biểu đồ cột kết quả RAGAS")

            # Kết nối các sự kiện huấn luyện và đánh giá
            run_ft_btn.click(
                fn=run_finetuning_ui,
                inputs=[llm_provider, api_key, sample_size, epochs, batch_size],
                outputs=[ft_status]
            )
            
            run_eval_btn.click(
                fn=run_evaluation_ui,
                inputs=[llm_provider, api_key],
                outputs=[eval_results_table, eval_chart]
            )

# Chạy ứng dụng nếu gọi trực tiếp file này
if __name__ == "__main__":
    demo.launch(share=True, css=custom_css, prevent_thread_lock=True)
