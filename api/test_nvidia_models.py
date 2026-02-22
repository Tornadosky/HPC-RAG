"""
Script to use NVIDIA AI API for embeddings and RAG operations - Docker Optimized
"""
import os
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

if not NVIDIA_API_KEY:
    print("Error: NVIDIA_API_KEY environment variable not set")
    exit(1)

print(f"Using NVIDIA API Key: {NVIDIA_API_KEY[:5]}...{NVIDIA_API_KEY[-5:]}")

try:
    # Import required dependencies
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    print("\nImporting necessary modules for RAG pipeline...")

    # Select embedding model from NVIDIA API
    EMBEDDING_MODEL = "nvidia/nv-embed-v1"  # This uses NVIDIA's API, no local models
    
    # Select LLM model from NVIDIA API
    LLM_MODEL = "meta/llama3-8b-instruct"  # This uses NVIDIA's API, no local models
    
    # Initialize the embeddings using NVIDIA's API
    print(f"\nInitializing NVIDIA API embeddings model: {EMBEDDING_MODEL}")
    embeddings = NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=NVIDIA_API_KEY
    )
    
    # Initialize the LLM using NVIDIA's API 
    print(f"Initializing NVIDIA API LLM model: {LLM_MODEL}")
    llm = ChatNVIDIA(
        model=LLM_MODEL,
        api_key=NVIDIA_API_KEY
    )
    
    # Define paths for text files - updated to use api/data/pdfs
    text_files_path = "api/data/pdfs/*.txt"
    text_files = glob.glob(text_files_path)
    
    if not text_files:
        print(f"Warning: No text files found at {text_files_path}")
        print("Please ensure your text files are in the api/data/pdfs directory with .txt extension")
    else:
        print(f"Found {len(text_files)} text files in api/data/pdfs directory")
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Process documents
        documents = []
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    file_name = os.path.basename(file_path)
                    chunks = text_splitter.split_text(content)
                    for i, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={"source": file_name, "chunk": i}
                        ))
                print(f"Processed {file_path} into {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        print(f"Created {len(documents)} document chunks in total")
        
        if documents:
            # Creating embeddings for documents - this uses NVIDIA API
            print("\nCreating embeddings for documents using NVIDIA API...")
            try:
                # Get embeddings for a sample document to test
                sample_doc = documents[0].page_content
                result = embeddings.embed_query(sample_doc)
                print(f"Successfully created embeddings using {EMBEDDING_MODEL}")
                print(f"Embedding vector length: {len(result)}")
                
                # Test RAG with a simple query
                print("\nTesting a simple RAG query against text files...")
                
                # Create a simple vector store with embeddings
                vector_store = FAISS.from_documents(documents, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                # Define a simple RAG prompt
                template = """You are an AI assistant for answering questions about HPC (High Performance Computing).
                Use the following retrieved context to answer the question. If you don't know the answer, just say you don't know.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:"""
                prompt = ChatPromptTemplate.from_template(template)
                
                # Create RAG chain
                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Test with a sample question
                question = "What are some popular HPC frameworks?"
                print(f"Question: {question}")
                response = rag_chain.invoke(question)
                print(f"Response: {response[:500]}...")
                
                # Export the vector store for reuse
                # Note: When loading this vector store, you'll need to use allow_dangerous_deserialization=True
                # since it uses pickle for serialization
                vector_store.save_local("api/vector_store_faiss")
                print("\nVector store saved to api/vector_store_faiss")
                
            except Exception as e:
                print(f"Error during RAG testing: {str(e)}")
        
except ImportError as e:
    print(f"Error importing required modules: {str(e)}")
    print("Please make sure you have installed the required packages:")
    print("pip install langchain-nvidia-ai-endpoints langchain-community langchain-core langchain-text-splitters faiss-cpu")
except Exception as e:
    print(f"General error: {str(e)}")

print("\nTest complete.") 