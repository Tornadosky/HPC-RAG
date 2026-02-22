# HPC RAG API with NVIDIA API Services

This project provides a Retrieval-Augmented Generation (RAG) API for High-Performance Computing (HPC) topics. It uses NVIDIA API services for embeddings and LLM capabilities without any local models.

## Features

- **API-Only Approach**: No local models - uses NVIDIA API services for all AI operations
- **Docker Optimized**: Lightweight container with only necessary dependencies
- **Persistent Vector Store**: Saves and reuses embeddings for efficiency
- **Multiple Model Support**: Can switch between different NVIDIA-hosted models
- **FastAPI Backend**: Clean, async API with automatic documentation

## Prerequisites

- Docker and Docker Compose
- NVIDIA API key
- Text files in `api/data/pdfs/` directory (with .txt extension)

## Getting Started

1. **Environment Setup**

   Create a `.env` file with your NVIDIA API key:
   ```
   NVIDIA_API_KEY=your_nvidia_api_key_here
   ```

2. **Add Text Files**

   Place your text files in the `api/data/pdfs/` directory with .txt extension. These will be processed and embedded for RAG functionality.

3. **Build and Run**

   From the api directory:
   ```bash
   docker-compose -f docker-compose.rag.yml up -d --build
   ```

   This will:
   - Build the Docker image
   - Process and embed your text files
   - Start the FastAPI server on port 8000

4. **Test the API**

   Once the container is running, you can test the API:
   ```bash
   python test_rag_api.py
   ```

   Or use the Swagger UI at `http://localhost:8000/docs`

## API Usage

### Query Endpoint

```
POST /query
```

Request body:
```json
{
  "query": "What is MPI and how is it used in HPC?",
  "model": "meta/llama3-8b-instruct"  // Optional, defaults to this model
}
```

Response:
```json
{
  "response": "MPI (Message Passing Interface) is a standardized and portable message-passing interface..."
}
```

## Available Models

You can use any of the NVIDIA-hosted LLM models. Some good options include:
- `meta/llama3-8b-instruct` (default, good balance of performance and speed)
- `meta/llama3-70b-instruct` (larger model, more capabilities)
- `nvidia/nemotron-4-mini-4b-instruct` (smaller model, faster responses)

## Customizing

### Change Embedding Model

To use a different embedding model, modify the `embedding_model` variable in `rag_api.py`.

### Adjust Chunk Size

For different document types, you may want to adjust the chunk size in `test_nvidia_models.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust this value
    chunk_overlap=100  # And this overlap
)
```

## Troubleshooting

- **API Not Starting**: Check Docker logs with `docker logs hpc-rag-backend`
- **No Vector Store**: Make sure text files exist in `api/data/pdfs/` and the container can access them
- **API Key Issues**: Ensure your NVIDIA API key is correctly set in the .env file
- **Deserialization Error**: If you see `ValueError: The de-serialization relies loading a pickle file...`, this is a safety feature in newer versions of LangChain. The code has been updated to handle this with `allow_dangerous_deserialization=True` since we're using our own trusted vector store.

## Security Note

The FAISS vector store uses pickle for serialization, which can be a security risk if loading untrusted files. Our implementation:

1. Only loads vector stores created by our own application
2. Uses the `allow_dangerous_deserialization=True` flag explicitly
3. Never loads vector stores from untrusted sources

This is safe for our use case, but be cautious if adapting this code for other applications. 