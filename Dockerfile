# movie_llm シーン検索デモ（GPU 24GB 以上。nvidia-container-toolkit が必要）
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf MOVIE_LLM_SAMPLES=/app/samples MOVIE_LLM_LORA=/app/models/timeloc-lora
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY movie_llm ./movie_llm
COPY scripts ./scripts
RUN python3 -m venv /venv && /venv/bin/pip install -U pip && /venv/bin/pip install -e ".[demo]"
ENV PATH=/venv/bin:$PATH
EXPOSE 7860
CMD ["python", "-m", "movie_llm.demo.app", "--host", "0.0.0.0", "--port", "7860"]
