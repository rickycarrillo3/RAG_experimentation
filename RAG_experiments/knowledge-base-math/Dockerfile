FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY extract.py ingest.py query.py app.py ./

# Create persistent volume mount points
RUN mkdir -p docs/raw docs/extracted chroma_db bm25_indexes

COPY startup.sh .
RUN chmod +x startup.sh

EXPOSE 7860

CMD ["./startup.sh"]
