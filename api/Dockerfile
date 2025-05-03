FROM python:3.12-slim

WORKDIR /app

# Install all required dependencies explicitly
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy scikit-learn joblib pydantic python-multipart

# Copy requirements.txt (for reference, we've installed dependencies above)
COPY requirements.txt .

COPY . .

# The model should be in the context when building
# We'll handle this in docker-compose with volumes

EXPOSE 8000

# Make sure we use the full path to uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 