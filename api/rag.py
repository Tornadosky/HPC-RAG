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

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Get environment variables with fallbacks
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")

class RAGSystem:
    def __init__(self):
        self.vectordb = None
        self.llm = None
        self.retriever = None
        self.embedder = None
        self.setup_embedding_model()
        self.setup_vector_db()
        self.setup_llm()
        
    def setup_embedding_model(self):
        """Set up the embedding model based on environment variables"""
        if EMBED_BASE_URL:
            # Use NVIDIA embeddings if base URL provided
            from langchain_community.embeddings import NVIDIAEmbeddings
            self.embedder = NVIDIAEmbeddings(
                base_url=EMBED_BASE_URL,
                model=EMBED_MODEL,
                api_key=NVIDIA_API_KEY
            )
        else:
            # Fall back to SentenceTransformer embeddings
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
            from langchain.chat_models import ChatOpenAI
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                base_url=LLM_BASE_URL,
                api_key=NVIDIA_API_KEY,
                temperature=0.1,
                streaming=True
            )
    
    def index_pdfs(self, pdf_directory: str):
        """Index all PDFs in the specified directory and load them into the vector store"""
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            print(f"PDF directory {pdf_directory} does not exist.")
            return False
        
        # Load PDFs from directory
        loader = DirectoryLoader(
            pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        print(f"Loading documents from {pdf_directory}...")
        documents = loader.load()
        if not documents:
            print("No documents found.")
            return False
        
        print(f"Loaded {len(documents)} documents.")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
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
    
    def create_prompt_template(self):
        """Create the chat prompt template"""
        system_template = """You are an HPC-programming advisor.
Context (cite as "(source.pdf)"):
{context}

User profile:
{profile}

Model ranking (top-3):
{ranking}

Question: {question}
Answer briefly (≤200 words) and cite sources."""

        human_template = "{question}"
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        return chat_prompt
    
    def format_ranking(self, ranking_list):
        """Format the ranking list for the prompt"""
        if not ranking_list:
            return "No model ranking available."
        
        formatted = []
        for rank in ranking_list[:3]:  # Take only top 3
            framework = rank.get("framework", "Unknown")
            prob = rank.get("prob", 0.0)
            formatted.append(f"{framework}: {prob:.4f}")
        
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
        # Create the chain
        prompt = self.create_prompt_template()
        
        # Convert profile to JSON string if it's not already a string
        if isinstance(profile, dict):
            profile_str = json.dumps(profile, indent=2)
        else:
            profile_str = profile
            
        # Format the ranking
        ranking_str = self.format_ranking(ranking)
        
        # Create the generation chain
        rag_chain = (
            {"context": self.retriever | self._format_docs, 
             "question": RunnablePassthrough(), 
             "profile": lambda _: profile_str,
             "ranking": lambda _: ranking_str}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        if stream:
            # Return a generator for streaming
            response_stream = rag_chain.stream(question)
            return response_stream
        else:
            # Return the complete response
            answer = rag_chain.invoke(question)
            citations = self.extract_citations(answer)
            return {"answer": answer, "citations": citations}

# Singleton instance
rag_system = None

def get_rag_system():
    """Get or create the RAG system singleton"""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system 