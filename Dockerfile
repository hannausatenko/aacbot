FROM python:3.11-slim

# System libs for cairosvg (cairo, pango, gdk-pixbuf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CPU-only PyTorch first to avoid pulling 2 GB of CUDA/NVIDIA packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY app.py llm.py rag.py ingest.py ./
COPY pictograms.db .
COPY data/ data/

# Bind to all interfaces so Docker port mapping works
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

# ANTHROPIC_API_KEY must be supplied at runtime via -e or --env-file
CMD ["python", "app.py"]
