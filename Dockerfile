# 10.200.99.202:15080/zero2x002/competition-base:ubuntu22.04-cuda12.3.2-cudnn9-py310.19
# 10.200.99.202:15080/zero2x002/competition-base:pytorch2.5.1-cuda12.1-cudnn9
FROM 10.200.99.202:15080/zero2x002/competition-base:ubuntu22.04-py310.19
WORKDIR /app

# 接收构建参数，指定 pip 缓存目录（默认值可留空）
ARG PIP_CACHE_DIR

# 将依赖文件单独复制（利用 Docker 层缓存）
COPY requirements.txt .

# 安装依赖，使用传入的缓存目录（如果 PIP_CACHE_DIR 有值，则使用 --cache-dir）
# 注意：如果 PIP_CACHE_DIR 为空，则仍会使用系统缓存（但无外部缓存时效果有限）
RUN pip install --cache-dir /tmp/pip-cache -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple


COPY . .

ENV FLASK_APP=train.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=30194"]