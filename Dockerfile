# 10.200.99.202:15080/zero2x002/competition-base:ubuntu22.04-cuda12.3.2-cudnn9-py310.19
# 10.200.99.202:15080/zero2x002/competition-base:pytorch2.5.1-cuda12.1-cudnn9
FROM 10.200.99.202:15080/zero2x002/competition-base:ubuntu22.04-py310.19
WORKDIR /workspace

ARG PIP_CACHE_DIR

COPY requirements.txt .
RUN pip install --cache-dir /tmp/pip-cache -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple

COPY . .

# models/ directory (with trained weights) is copied in via COPY . .
# Ensure it exists even if empty so inference falls back to placeholder
RUN mkdir -p /workspace/models

CMD ["bash", "run.sh"]