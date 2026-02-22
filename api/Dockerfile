FROM python:3.10-slim AS builder

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create directories
RUN mkdir -p api/data/pdfs api/vector_store_faiss

# Copy necessary files
COPY api/test_nvidia_models.py ./api/
COPY api/main.py ./api/
COPY api/rag.py ./api/
COPY model.joblib ./
COPY api/data/pdfs/*.txt ./api/data/pdfs/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Default command - first run the test to create vector store, then start API
CMD ["sh", "-c", "python api/test_nvidia_models.py && python api/main.py"] 