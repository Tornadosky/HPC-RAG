FROM python:3.12-slim

WORKDIR /app

# Copy requirements.txt first for better caching
COPY api/requirements.txt .

# Install all required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY api/ .

# Copy model file
COPY model.joblib .

# Create necessary directories
RUN mkdir -p ./data/pdfs ./data/chroma_db

EXPOSE 8000

# Make sure we use the full path to uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 