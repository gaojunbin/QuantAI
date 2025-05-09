FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建数据目录
RUN mkdir -p data/predictions

# 设置环境变量
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "src/realtime/web_app.py"] 