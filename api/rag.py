import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain imports
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Hugging Face setup
import huggingface_hub
from huggingface_hub import login

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Get environment variables with fallbacks
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")

# Login to Hugging Face if token is provided
if HUGGINGFACE_API_KEY:
    print(f"Logging into Hugging Face with provided token")
    try:
        login(token=HUGGINGFACE_API_KEY)
        print("Successfully logged into Hugging Face")
    except Exception as e:
        print(f"Error logging into Hugging Face: {str(e)}")

class RAGSystem:
    def __init__(self):
        self.vectordb = None
        self.llm = None
        self.retriever = None
        self.embedder = None
        self.print_available_nvidia_models()
        self.setup_embedding_model()
        self.setup_vector_db()
        self.setup_llm()
        
    def print_available_nvidia_models(self):
        """Print available NVIDIA AI models for debugging"""
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            print("Checking available NVIDIA models...")
            available_models = ChatNVIDIA.get_available_models(api_key=NVIDIA_API_KEY)
            print("Available NVIDIA LLM models:")
            for model in available_models:
                print(f"  - {model}")
                
            # Try to get available embedding models if possible
            try:
                from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
                embed_models = NVIDIAEmbeddings.get_available_models(api_key=NVIDIA_API_KEY)
                print("Available NVIDIA Embedding models:")
                for model in embed_models:
                    print(f"  - {model}")
            except Exception as e:
                print(f"Could not get available embedding models: {str(e)}")
        except Exception as e:
            print(f"Error getting available models: {str(e)}")
            
    def setup_embedding_model(self):
        """Set up the embedding model based on environment variables"""
        print(f"Setting up embedding model: {EMBED_MODEL}")
        
        # Check if we're using Nvidia embeddings
        if EMBED_MODEL == "nvidia/nv-embed-v1" and HUGGINGFACE_API_KEY:
            # Use HuggingFace embeddings for nv-embed-v1
            from langchain.embeddings import HuggingFaceEmbeddings
            print(f"Using HuggingFace embeddings with model: {EMBED_MODEL}")
            self.embedder = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"trust_remote_code": True}
            )
        elif EMBED_MODEL.startswith("nvidia/"):
            # Use NVIDIA API embeddings for other Nvidia models
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
            print(f"Using NVIDIA AI Endpoints embeddings with model: {EMBED_MODEL}")
            self.embedder = NVIDIAEmbeddings(
                model=EMBED_MODEL,
                api_key=NVIDIA_API_KEY
            )
        else:
            # Use SentenceTransformer embeddings for non-Nvidia models
            print(f"Using SentenceTransformer embeddings with model: {EMBED_MODEL}")
            self.embedder = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    
    def setup_vector_db(self):
        """Set up the vector database"""
        # Create the Chroma directory if it doesn't exist
        chroma_dir = Path(CHROMA_PATH)
        chroma_dir.parent.mkdir(parents=True, exist_ok=True)
        chroma_dir.mkdir(exist_ok=True)
        
        self.vectordb = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=self.embedder
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 6})
    
    def setup_llm(self):
        """Set up the LLM model based on environment variables"""
        print(f"Setting up LLM with base URL: {LLM_BASE_URL}, model: {LLM_MODEL}")
        
        # If we're using localhost, assume Ollama
        if "localhost" in LLM_BASE_URL or "ollama" in LLM_BASE_URL:
            self.llm = ChatOllama(
                model=LLM_MODEL,
                base_url=LLM_BASE_URL.replace("/v1", ""),
                temperature=0.1,
                streaming=True
            )
        else:
            # For production with NVIDIA NIM
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            print(f"Using NVIDIA NIM API with model: {LLM_MODEL}")
            
            # Check if model has the full path format (with provider)
            if "/" not in LLM_MODEL:
                # For older models that might not have provider prefix
                if LLM_MODEL in ["llama3", "llama3-8b", "llama3-70b"]:
                    model_name = f"meta/{LLM_MODEL}-instruct"
                else:
                    model_name = LLM_MODEL
            else:
                model_name = LLM_MODEL
                
            print(f"Using model name: {model_name}")
            
            self.llm = ChatNVIDIA(
                model=model_name,
                api_key=NVIDIA_API_KEY,
                temperature=0.1,
                streaming=True
            )
            
    def create_output_parser(self):
        """Create a structured output parser for formatted responses"""
        response_schemas = [
            ResponseSchema(
                name="answer",
                description="The answer to the user's question based on the retrieved documents, user profile and ranking.",
                type="string"
            ),
            ResponseSchema(
                name="sources",
                description="List of source documents cited in the answer.",
                type="array"
            )
        ]
        
        return StructuredOutputParser.from_response_schemas(response_schemas)
    
    def index_pdfs(self, pdf_directory: str):
        """Index all PDFs and TXT files in the specified directory and load them into the vector store"""
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            print(f"PDF directory {pdf_directory} does not exist.")
            return False
        
        all_documents = []
        
        # Load PDFs from directory
        pdf_loader = DirectoryLoader(
            pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        print(f"Loading PDF documents from {pdf_directory}...")
        pdf_documents = pdf_loader.load()
        if pdf_documents:
            print(f"Loaded {len(pdf_documents)} PDF documents.")
            all_documents.extend(pdf_documents)
        else:
            print("No PDF documents found.")
        
        # Load TXT files from directory
        from langchain.document_loaders import TextLoader
        txt_loader = DirectoryLoader(
            pdf_directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        
        print(f"Loading TXT documents from {pdf_directory}...")
        txt_documents = txt_loader.load()
        if txt_documents:
            print(f"Loaded {len(txt_documents)} TXT documents.")
            all_documents.extend(txt_documents)
        else:
            print("No TXT documents found.")
        
        if not all_documents:
            print("No documents found in total.")
            return False
        
        print(f"Loaded a total of {len(all_documents)} documents.")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(all_documents)
        
        print(f"Split into {len(chunks)} chunks.")
        
        # Add source and page metadata to each chunk
        for chunk in chunks:
            source_file = Path(chunk.metadata["source"]).name
            chunk.metadata["source"] = source_file
            # Add a default rf_rank of 0 if not available
            if "rf_rank" not in chunk.metadata:
                chunk.metadata["rf_rank"] = 0
        
        # Add to vector store
        self.vectordb.add_documents(chunks)
        self.vectordb.persist()
        print(f"Indexed {len(chunks)} chunks and persisted to {CHROMA_PATH}")
        return True
    
    def _format_docs(self, docs):
        """Format the documents for the prompt context"""
        return "\n\n".join(f"Document from {doc.metadata['source']}, Page {doc.metadata.get('page', 'N/A')}:\n{doc.page_content}" for doc in docs)
    
    def create_prompt_template(self, use_structured_output=False):
        """Create the chat prompt template with optional structured output format instructions"""
        base_system_template = """You are an HPC-programming advisor who helps users understand framework recommendations.

Context from the literature (cite as "(source.pdf)"):
{context}

User's hardware profile and preferences:
{profile}

Framework ranking (from most to least suitable according to your profile):
{ranking}

When responding:
1. Focus on providing educational and helpful information about HPC frameworks
2. Present probability scores as percentages (e.g., write 6.7% instead of 0.067)
3. Explain why frameworks are ranked as they are based on the user's profile and needs
4. Use your general knowledge about HPC frameworks when appropriate 
5. Keep framework names properly capitalized (e.g., OpenMP, ALPAKA, SYCL)
6. Ensure proper spacing around punctuation marks and between words
7. Keep answers concise (≤200 words) and talk as if you are talking to user, (e.g. "Based on your profile, you should use OpenMP because...")
8. Do NOT mention internal variable names (like need_cross_vendor, domain_hpc, etc.)
9. For sources, use short descriptions like ("As referenced by some resources") instead of document names
10. IMPORTANT: Do not add spaces within words (e.g., write "prioritizes" not "priorit izes")

Language guidelines:
- Maintain proper capitalization for framework names (OpenMP, CUDA, ALPAKA, etc.)
- Avoid inserting spaces into words, acronyms or framework names
- Write percentages without spaces (6.7% not 6.7 %)
- Use proper spacing after punctuation, but not before
- Format apostrophes correctly (user's not user 's)
- Use plain language to explain technical concepts
- Shorten long source filenames to 15 characters with "..." in citations
- Keep words joined correctly (write "functionality" not "function ality")
"""

        if use_structured_output:
            # Add structured output format instructions
            output_parser = self.create_output_parser()
            format_instructions = output_parser.get_format_instructions()
            system_template = f"{base_system_template}\n\n{format_instructions}\n\nQuestion: {{question}}"
            return ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}")
            ]), output_parser
        else:
            # Standard text output
            system_template = f"{base_system_template}\n\nQuestion: {{question}}"
            return ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}")
            ]), None
    
    def format_ranking(self, ranking_list):
        """Format the ranking list for the prompt"""
        if not ranking_list:
            return "No model ranking available."
        
        formatted = []
        for rank, item in enumerate(ranking_list[:3], 1):  # Take only top 3
            framework = item.get("framework", "Unknown")
            prob = item.get("prob", 0.0)
            percentage = prob * 100
            formatted.append(f"{rank}. {framework}: {percentage:.1f}%")
        
        return "\n".join(formatted)
    
    def extract_citations(self, answer):
        """Extract citation sources from the answer"""
        import re
        # Find all patterns like (source.pdf)
        citations = re.findall(r'\([^)]*\.pdf\)', answer)
        # Clean up citations and remove duplicates
        clean_citations = []
        for citation in citations:
            clean = citation.strip('()')
            if clean not in clean_citations:
                clean_citations.append(clean)
        return clean_citations
    
    def query(self, question, profile, ranking, stream=False):
        """Query the RAG system with a user question"""
        # Create the chain with structured output when not streaming
        use_structured_output = not stream
        prompt, output_parser = self.create_prompt_template(use_structured_output)
        
        # Convert profile to JSON string if it's not already a string
        if isinstance(profile, dict):
            profile_str = json.dumps(profile, indent=2)
        else:
            profile_str = profile
            
        # Format the ranking
        ranking_str = self.format_ranking(ranking)
        
        # Input variables
        input_variables = {
            "context": self.retriever | self._format_docs, 
            "question": RunnablePassthrough(), 
            "profile": lambda _: profile_str,
            "ranking": lambda _: ranking_str
        }
        
        if stream:
            # For streaming, we use the standard chain without structured output
            rag_chain = (
                input_variables 
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Return a generator for streaming
            response_stream = rag_chain.stream(question)
            return response_stream
        else:
            # For non-streaming, use structured output
            rag_chain = (
                input_variables
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Get the raw response
            raw_response = rag_chain.invoke(question)
            
            try:
                # Parse the structured output
                if output_parser:
                    parsed_output = output_parser.parse(raw_response)
                    # If the model returned structured output correctly
                    if isinstance(parsed_output, dict) and "answer" in parsed_output:
                        return {
                            "answer": parsed_output["answer"],
                            "citations": parsed_output.get("sources", [])
                        }
                
                # Fallback if structured parsing fails
                citations = self.extract_citations(raw_response)
                return {"answer": raw_response, "citations": citations}
            except Exception as e:
                print(f"Error parsing structured output: {str(e)}")
                # Fallback to regular extraction
                citations = self.extract_citations(raw_response)
                return {"answer": raw_response, "citations": citations}

# Singleton instance
rag_system = None

def get_rag_system():
    """Get or create the RAG system singleton"""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system 