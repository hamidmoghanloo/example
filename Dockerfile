FROM docker.arvancloud.ir/pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch; print('PyTorch:', torch.__version__); assert torch.__version__.startswith('2.6.0'), 'Expected PyTorch 2.6.0 from base image'"

COPY app.py ./

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
