import json
import re
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel

class QueryRouter:
    """
    Giai đoạn 2 (GD 2: Routing & Transformation):
    - Router siêu nhẹ giúp phân tích ý định (Intent) của người dùng.
    - Phân loại câu hỏi thành:
      1. 'local': Các câu hỏi liên quan đến kiến thức lưu trữ (AIO course, luật Việt Nam).
      2. 'web': Các câu hỏi cần tin tức thời sự cập nhật, thời tiết, hoặc các sự kiện bên ngoài.
      3. 'hybrid': Cần kết hợp tra cứu tài liệu và đối chiếu thông tin cập nhật trên Internet.
    - Ép định dạng JSON để tiết kiệm Output Token tối đa (tránh AI giải thích dông dài).
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        
        # Prompt siêu ngắn, tối ưu hóa kích thước Input Token
        self.prompt_template = """Phân loại câu hỏi của người dùng vào một trong 3 nhóm nguồn thông tin:
- "local": Tài liệu chuyên ngành về AI, lập trình, hoặc các bộ Luật Lao động, Luật Đất đai Việt Nam.
- "web": Câu hỏi về tin tức thời sự cập nhật, thời tiết hôm nay, sự kiện mới diễn ra hoặc thông tin đời sống chung.
- "hybrid": Câu hỏi cần đối chiếu dữ liệu nội bộ đồng thời kiểm tra thêm thông tin cập nhật ngoài internet.

CHỈ trả về kết quả dưới định dạng JSON duy nhất như sau, tuyệt đối không viết thêm bất kỳ lời giải thích nào:
{{"route": "local" | "web" | "hybrid"}}

Câu hỏi của User: "{query}"
JSON Output:"""
        
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        
    def route_query(self, query: str) -> str:
        """
        Định tuyến câu hỏi và trả về chuỗi 'local', 'web' hoặc 'hybrid'.
        """
        formatted_prompt = self.prompt.format(query=query)
        
        try:
            # Gọi LLM sinh phản hồi
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content.strip()
            
            # Loại bỏ các ký tự bọc markdown json nếu có
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Phân tích cú pháp JSON
            data = json.loads(response_text)
            route = data.get("route", "local").lower().strip()
            
            if route in ["local", "web", "hybrid"]:
                return route
            return "local" # Fallback mặc định an toàn
            
        except Exception as e:
            # Fallback dự phòng bằng Regex nếu JSON lỗi hoặc LLM không tuân thủ định dạng
            print(f"[Cảnh báo] Lỗi parse JSON ở Router: {str(e)}. Sử dụng Regex fallback.")
            text = response.content.strip() if 'response' in locals() else ""
            if "local" in text.lower():
                return "local"
            elif "web" in text.lower():
                return "web"
            elif "hybrid" in text.lower():
                return "hybrid"
            return "local" # Fallback mặc định
