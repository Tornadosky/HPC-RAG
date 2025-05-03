FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY streamlit_rag.py /app/
RUN pip install --no-cache-dir streamlit requests

# Expose the Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "streamlit_rag.py", "--server.port=8501", "--server.address=0.0.0.0"] 