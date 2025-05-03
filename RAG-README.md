# HPC Framework Recommender with RAG

This extension adds a Retrieval-Augmented Generation (RAG) layer to the HPC Framework Recommender, allowing users to ask detailed questions about HPC frameworks and get accurate, contextual answers based on research papers and expert knowledge.

## Features

- **LLM Integration**: Uses Meta's Llama-3-8B-Instruct model served via Ollama (dev) or NVIDIA NIM (prod)
- **Vector Database**: Indexes research papers into a Chroma vector store for semantic search
- **Contextual Answers**: Combines user profile, framework rankings, and retrieved document chunks for precise responses
- **Citation Support**: Automatically cites sources used to generate answers
- **Real-time Streaming**: Supports streaming responses for a better user experience
- **Streamlit Demo**: Includes a standalone demo application

## Architecture

The RAG system works as follows:

1. **Document Indexing**:
   - PDF research papers are loaded, chunked, and indexed into a Chroma vector store
   - Each chunk maintains metadata (source, page, etc.)

2. **Query Processing**:
   - User questions are processed to retrieve relevant document chunks
   - The user's profile and framework rankings are included in the prompt context
   - A carefully crafted prompt template guides the LLM to generate precise answers

3. **Response Generation**:
   - The LLM generates responses based on retrieved information and citations
   - Answers are capped at 200 words for conciseness
   - Citations are tracked and included in the response

## Environment Variables

The system is configurable through the following environment variables:

- `LLM_BASE_URL`: Base URL for the LLM API (default: `http://localhost:11434/v1` for Ollama)
- `LLM_MODEL`: Model name to use (default: `llama3:8b-instruct`)
- `EMBED_BASE_URL`: Base URL for embeddings API (if using NVIDIA embeddings)
- `EMBED_MODEL`: Embedding model to use (default: `all-MiniLM-L6-v2`)
- `NVIDIA_API_KEY`: API key for NVIDIA NIM (if applicable)
- `CHROMA_PATH`: Path to store Chroma vector database (default: `./data/chroma_db`)

## API Endpoints

### Chat Endpoint

```
POST /chat
```

Request Body:
```json
{
  "question": "What makes SYCL different from OpenCL?",
  "profile": {
    "hw_nvidia": 1,
    "perf_weight": 0.8,
    "port_weight": 0.7,
    ...
  },
  "ranking": [
    {"framework": "CUDA", "prob": 0.75},
    {"framework": "OpenMP", "prob": 0.15},
    {"framework": "SYCL", "prob": 0.10}
  ]
}
```

Response:
```json
{
  "answer": "SYCL differs from OpenCL by providing a higher-level, single-source programming model...",
  "citations": ["sycl_overview.pdf", "opencl_comparison.pdf"]
}
```

For streaming responses, add `?stream=true` query parameter.

### PDF Indexing Endpoint

```
POST /index-pdfs
```

Starts an asynchronous process to index PDFs in the configured directory.

## Running the System

### Development Environment

1. **Prerequisites**:
   - Docker and Docker Compose
   - NVIDIA GPU with appropriate drivers (for optimal performance)

2. **Setup**:
   ```bash
   # Initialize the RAG environment
   chmod +x init_rag.sh
   ./init_rag.sh
   
   # Start the full stack
   docker-compose up -d
   ```

3. **Adding Research Papers**:
   - Place PDF files in the `api/data/pdfs` directory
   - Trigger indexing: `curl -X POST http://localhost:8000/index-pdfs`
   - Alternatively, create a `test_pdfs` directory and use the init script

### Accessing the Applications

- **Streamlit Demo**: http://localhost/rag/
- **API**: http://localhost/api/
- **Frontend**: http://localhost/

## Streamlit Demo Usage

1. Fill out your HPC profile in the sidebar
2. Click "Get Recommendations" to see framework rankings
3. Use the chat interface to ask questions about the recommended frameworks
4. View sources used by expanding the "Sources" section in responses

## Production Deployment

For production deployments:

1. Set up NVIDIA NIM for serving the Llama-3 model
2. Configure environment variables for the API service
3. Ensure proper security measures (API keys, restricted access, etc.)
4. Use a production-ready document storage solution for PDFs

## License

Same as the main project. 