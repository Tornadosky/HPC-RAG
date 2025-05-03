#!/bin/bash

# Script to initialize the RAG environment

# Create necessary directories
mkdir -p api/data/pdfs
mkdir -p api/data/chroma_db

# Start Ollama (if not already running)
if ! docker ps | grep -q ollama; then
    echo "Starting Ollama container..."
    docker-compose up -d ollama
    # Give Ollama some time to start up
    sleep 10
fi

# Pull the Llama model
echo "Pulling Llama 3 8B Instruct model (this may take a while)..."
docker exec -it $(docker ps -qf "name=ollama") ollama pull llama3:8b-instruct

echo "Model pulled successfully."

# Add test PDF files (if any exist)
if [ -d "test_pdfs" ]; then
    echo "Copying test PDFs to the data directory..."
    cp -r test_pdfs/* api/data/pdfs/
    echo "Test PDFs copied successfully."
fi

# Index PDFs if API is running
if docker ps | grep -q api; then
    echo "Triggering PDF indexing..."
    curl -X POST http://localhost:8000/index-pdfs
    echo "PDF indexing started."
else
    echo "API container is not running. Start it with docker-compose up -d api"
    echo "Then trigger PDF indexing manually with: curl -X POST http://localhost:8000/index-pdfs"
fi

echo "Initialization complete!"
echo "- Start the full stack with: docker-compose up -d"
echo "- Access the Streamlit demo at: http://localhost/rag/"
echo "- Access the API at: http://localhost/api/"
echo "- Access the Frontend at: http://localhost/" 