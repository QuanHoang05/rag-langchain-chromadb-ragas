import re
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel

class HyDEQueryTransformer:
    """
    Kỹ thuật Hypothetical Document Embeddings (HyDE):
    - Yêu cầu LLM viết một câu trả lời giả định (Draft/Hypothetical Answer) cho câu hỏi.
    - Chuyển câu hỏi ở dạng "nghi vấn" thành câu trả lời ở dạng "khẳng định/mô tả" giúp so khớp vector tốt hơn.
    - Giới hạn độ dài output (max 100-150 từ) để tránh lãng phí tokens.
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt_template = """Hãy viết một đoạn văn ngắn (khoảng 100 từ) trả lời giả định cho câu hỏi dưới đây.
Viết như thể đây là một đoạn trích dẫn chuyên môn từ sách giáo khoa hoặc tài liệu hướng dẫn.
Không cần bắt đầu bằng những câu như "Theo tôi...", "Câu trả lời giả định...", hãy đi thẳng vào nội dung chính.

Câu hỏi: {question}
Đoạn văn giả định:"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        
    def generate_hypothetical_doc(self, question: str) -> str:
        formatted_prompt = self.prompt.format(question=question)
        try:
            # Ép trần tokens đầu ra bằng cách gọi LLM thông thường
            response = self.llm.invoke(formatted_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"[Cảnh báo] Lỗi khi tạo tài liệu giả định HyDE: {str(e)}")
            return question  # Fallback: trả về câu hỏi gốc nếu lỗi


class DecompositionQueryTransformer:
    """
    Kỹ thuật Query Decomposition (Phân rã câu hỏi):
    - Tách một câu hỏi phức tạp hoặc câu hỏi chứa nhiều ý so sánh thành 2-3 câu hỏi con đơn giản hơn.
    - Giới hạn cứng tối đa 3 câu hỏi con để tiết kiệm chi phí gọi mô hình và bảo vệ token.
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt_template = """Hãy phân tích và tách câu hỏi gốc dưới đây thành tối đa 3 câu hỏi phụ (sub-queries) đơn giản, mỗi câu tập trung vào một khía cạnh riêng biệt.
Nếu câu hỏi đã đơn giản, chỉ cần trả về chính câu hỏi đó.
Mỗi câu hỏi phụ viết trên một dòng mới. Không viết số thứ tự, không kèm ký tự đặc biệt ở đầu dòng.

Câu hỏi gốc: {question}
Các câu hỏi phụ:"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        
    def decompose(self, question: str) -> List[str]:
        formatted_prompt = self.prompt.format(question=question)
        try:
            response = self.llm.invoke(formatted_prompt)
            lines = response.content.strip().split("\n")
            
            sub_queries = []
            for line in lines:
                cleaned_line = line.strip()
                # Loại bỏ các đầu mục số dạng "1.", "-", "*", "•"
                cleaned_line = re.sub(r"^[\d\.\-\*\•\s]+", "", cleaned_line).strip()
                if cleaned_line and len(cleaned_line) > 5:
                    sub_queries.append(cleaned_line)
                    
            # Giới hạn trần tối đa 3 câu hỏi phụ để tránh bùng nổ token
            return sub_queries[:3]
        except Exception as e:
            print(f"[Cảnh báo] Lỗi phân rã câu hỏi: {str(e)}")
            return [question]  # Fallback: trả về danh sách chỉ chứa câu hỏi gốc
