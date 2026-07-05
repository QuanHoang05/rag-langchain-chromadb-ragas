# Sử dụng Python 3.10 slim làm base image
FROM python:3.10-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Cài đặt các công cụ hệ thống cần thiết (cho việc biên dịch và các thư viện C++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libsqlite3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements trước để tận dụng cache của Docker
COPY requirements.txt .

# Nâng cấp pip và cài đặt các thư viện Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Khai báo biến môi trường cho Gradio chạy ở chế độ docker
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Expose cổng của ứng dụng Gradio
EXPOSE 7860

# Lệnh chạy ứng dụng khi container khởi động
CMD ["python", "app.py"]
