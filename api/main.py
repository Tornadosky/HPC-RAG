from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import os
import pathlib
from sse_starlette.sse import EventSourceResponse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

# Import RAG functionality
from rag import get_rag_system

# Import NVIDIA API dependencies
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

# NVIDIA API key for RAG
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Global variables for NVIDIA RAG
nvidia_embedding_model = "nvidia/nv-embed-v1"
nvidia_vector_store_path = "api/vector_store_faiss"
nvidia_embeddings = None
nvidia_vector_store = None
nvidia_llm = None

# Lifespan context manager for app startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize NVIDIA RAG components if API key is provided
    if NVIDIA_API_KEY:
        await initialize_nvidia_rag()
    
    # App runs here
    yield
    
    # Shutdown: cleanup resources
    print("Shutting down API")

# Function to initialize NVIDIA RAG components
async def initialize_nvidia_rag():
    global nvidia_embeddings, nvidia_vector_store, nvidia_llm
    
    try:
        print(f"Initializing NVIDIA API embeddings model: {nvidia_embedding_model}")
        nvidia_embeddings = NVIDIAEmbeddings(
            model=nvidia_embedding_model,
            api_key=NVIDIA_API_KEY
        )
        
        # Check if vector store exists
        if os.path.exists(nvidia_vector_store_path):
            print(f"Loading vector store from {nvidia_vector_store_path}")
            nvidia_vector_store = FAISS.load_local(
                nvidia_vector_store_path, 
                nvidia_embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Initialize default LLM
            nvidia_llm = ChatNVIDIA(
                model="meta/llama3-8b-instruct",
                api_key=NVIDIA_API_KEY
            )
            
            print("NVIDIA RAG initialization complete")
        else:
            print(f"Warning: Vector store not found at {nvidia_vector_store_path}. NVIDIA RAG will not be available.")
    except Exception as e:
        print(f"Error initializing NVIDIA RAG: {str(e)}")

app = FastAPI(
    title="HPC Framework Recommender API", 
    description="API for recommending HPC frameworks based on user preferences",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input schema
class FrameworkRequest(BaseModel):
    hw_cpu: int = Field(0, description="Has CPU")
    hw_nvidia: int = Field(0, description="Has NVIDIA GPU")
    hw_amd: int = Field(0, description="Has AMD GPU")
    hw_other: int = Field(0, description="Has other accelerators")
    need_cross_vendor: int = Field(0, description="Needs cross-vendor support")
    perf_weight: float = Field(0.5, description="Weight given to performance")
    port_weight: float = Field(0.5, description="Weight given to portability")
    eco_weight: float = Field(0.5, description="Weight given to ecosystem")
    pref_directives: int = Field(0, description="Prefers directive-based programming")
    pref_kernels: int = Field(0, description="Prefers kernel-based programming")
    greenfield: int = Field(0, description="New project")
    gpu_extend: int = Field(0, description="Extending existing GPU code")
    cpu_port: int = Field(0, description="Porting from CPU code")
    domain_ai_ml: int = Field(0, description="AI/ML domain")
    domain_hpc: int = Field(0, description="HPC domain")
    domain_climate: int = Field(0, description="Climate domain")
    domain_embedded: int = Field(0, description="Embedded domain")
    domain_graphics: int = Field(0, description="Graphics domain")
    domain_data_analytics: int = Field(0, description="Data analytics domain")
    domain_other: int = Field(0, description="Other domain")
    lockin_tolerance: float = Field(0.5, description="Tolerance for vendor lock-in")
    gpu_skill_level: int = Field(1, description="GPU programming skill level")

# Define the response schema
class FrameworkRanking(BaseModel):
    framework: str
    prob: float

class PredictionResponse(BaseModel):
    ranking: List[FrameworkRanking]
    explanation: str

# New models for the RAG chat endpoint
class ChatRequest(BaseModel):
    question: str
    profile: Dict[str, Any]
    ranking: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    answer: str
    citations: List[str]

# New models for NVIDIA RAG endpoint
class NvidiaRagRequest(BaseModel):
    query: str
    model: Optional[str] = "meta/llama3-8b-instruct"
    framework_ranking: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    user_profile: Optional[Dict[str, Any]] = Field(default_factory=dict)

class NvidiaRagResponse(BaseModel):
    response: str

# Find the model file (search in parent directory if not in current directory)
def find_model_file():
    model_filename = "model.joblib"
    current_dir = pathlib.Path(__file__).parent.absolute()
    
    # Try current directory
    model_path = current_dir / model_filename
    print(f"Looking for model at: {model_path}")
    if model_path.exists():
        print(f"Found model at: {model_path}")
        return model_path
    
    # Try parent directory
    parent_model_path = current_dir.parent / model_filename
    print(f"Looking for model at parent: {parent_model_path}")
    if parent_model_path.exists():
        print(f"Found model at: {parent_model_path}")
        return parent_model_path
    
    # Try root directory (Docker volume)
    root_model_path = pathlib.Path("/app") / model_filename
    print(f"Looking for model at root: {root_model_path}")
    if root_model_path.exists():
        print(f"Found model at: {root_model_path}")
        return root_model_path
        
    print(f"Model not found in any location")
    return None

# Load the pre-trained model
def load_model():
    try:
        model_path = find_model_file()
        if model_path is None:
            raise FileNotFoundError(f"Model file not found. Please ensure model.joblib exists.")
        
        print(f"Loading model from: {model_path}")
        model_data = joblib.load(model_path)
        print(f"Model loaded successfully!")
        return model_data
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Error loading model: {str(e)}")

# Generate explanation based on input features and prediction
def generate_explanation(input_data, top_framework, probabilities, classes):
    explanation_parts = []
    
    # Check if any hardware is selected and add to explanation
    hardware_used = []
    if input_data["hw_nvidia"] == 1:
        hardware_used.append("NVIDIA GPU")
    if input_data["hw_amd"] == 1:
        hardware_used.append("AMD GPU")
    if input_data["hw_cpu"] == 1:
        hardware_used.append("CPU")
    if input_data["hw_other"] == 1:
        hardware_used.append("other accelerators")
    
    # Identify most important preferences based on weights
    preferences = []
    if input_data["perf_weight"] > 0.7:
        preferences.append("high performance")
    if input_data["port_weight"] > 0.7:
        preferences.append("portability")
    if input_data["eco_weight"] > 0.7:
        preferences.append("mature ecosystem")
    
    # Check programming model preference
    prog_model = []
    if input_data["pref_directives"] == 1:
        prog_model.append("directive-based programming")
    if input_data["pref_kernels"] == 1:
        prog_model.append("kernel-based programming")
    
    # Check project type
    project_type = None
    if input_data["greenfield"] == 1:
        project_type = "new project"
    elif input_data["gpu_extend"] == 1:
        project_type = "extending existing GPU code"
    elif input_data["cpu_port"] == 1:
        project_type = "porting from CPU code"
    
    # Combine into explanation
    if hardware_used:
        explanation_parts.append(f"your hardware ({', '.join(hardware_used)})")
    
    if preferences:
        explanation_parts.append(f"your preference for {', '.join(preferences)}")
    
    if prog_model:
        explanation_parts.append(f"your preference for {', '.join(prog_model)}")
    
    if project_type:
        explanation_parts.append(f"your {project_type}")
    
    if input_data["need_cross_vendor"] == 1:
        explanation_parts.append("your need for cross-vendor support")
    
    if input_data["lockin_tolerance"] > 0.8:
        explanation_parts.append("your high tolerance for vendor lock-in")
    elif input_data["lockin_tolerance"] < 0.3:
        explanation_parts.append("your preference for vendor independence")
    
    # Check domain focus
    domains = []
    if input_data["domain_hpc"] == 1:
        domains.append("HPC")
    if input_data["domain_ai_ml"] == 1:
        domains.append("AI/ML")
    if input_data["domain_climate"] == 1:
        domains.append("climate simulation")
    if input_data["domain_embedded"] == 1:
        domains.append("embedded systems")
    if input_data["domain_graphics"] == 1:
        domains.append("graphics")
    if input_data["domain_data_analytics"] == 1:
        domains.append("data analytics")
        
    if domains:
        explanation_parts.append(f"your focus on {', '.join(domains)}")
    
    # Construct the explanation
    if explanation_parts:
        explanation = f"Based on {', '.join(explanation_parts)}, {top_framework} is your best match."
    else:
        explanation = f"{top_framework} is recommended based on your input criteria."
    
    return explanation

@app.get("/")
async def root():
    return {"message": "HPC Framework Recommender API is running. Use /predict to get recommendations and /chat for RAG-assisted responses."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: FrameworkRequest):
    try:
        print(f"Received prediction request with data: {request}")
        
        # Load the model
        print("Loading model...")
        model_data = load_model()
        if not model_data or len(model_data) != 2:
            print(f"Invalid model data: {model_data}")
            raise HTTPException(status_code=500, detail="Invalid model data")
        
        calibrated_rf, scaler = model_data
        
        # Convert request to DataFrame
        input_dict = request.dict()
        input_df = pd.DataFrame([input_dict])
        print(f"Input dataframe created with shape: {input_df.shape}")
        print(f"Input data: {input_dict}")
        
        # Scale the data
        try:
            input_scaled = scaler.transform(input_df)
            print(f"Data scaled successfully")
        except Exception as e:
            print(f"Error scaling data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error scaling data: {str(e)}")
        
        # Make predictions
        try:
            probabilities = calibrated_rf.predict_proba(input_scaled)[0]
            class_names = calibrated_rf.classes_
            print(f"Prediction successful.")
            print(f"Classes: {class_names}")
            print(f"Raw probabilities: {probabilities}")
            print(f"Probabilities as percentages: {[round(p * 100, 2) for p in probabilities]}")
            
            # Print sorted probabilities for better visibility
            sorted_probs = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)
            print(f"Sorted predictions:")
            for framework, prob in sorted_probs:
                print(f"  {framework}: {prob:.4f} ({prob * 100:.2f}%)")
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
        
        # Create ranking
        ranking = [
            FrameworkRanking(framework=framework, prob=float(prob))
            for framework, prob in zip(class_names, probabilities)
        ]
        
        # Sort by probability (descending)
        ranking.sort(key=lambda x: x.prob, reverse=True)
        
        # Generate explanation
        top_framework = ranking[0].framework
        explanation = generate_explanation(input_dict, top_framework, probabilities, class_names)
        
        print(f"Successfully generated prediction with top framework: {top_framework}")
        return PredictionResponse(ranking=ranking, explanation=explanation)
    
    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Unexpected error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/index-pdfs")
async def index_pdfs(background_tasks: BackgroundTasks):
    """Endpoint to trigger PDF indexing (async task)"""
    try:
        # Get the RAG system
        rag_system = get_rag_system()
        
        # Define the PDF directory
        pdf_directory = os.path.join(os.path.dirname(__file__), "data", "pdfs")
        
        # Queue the indexing task in the background
        background_tasks.add_task(rag_system.index_pdfs, pdf_directory)
        
        return {"message": f"PDF indexing started in the background for directory: {pdf_directory}"}
    except Exception as e:
        print(f"Error starting PDF indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting PDF indexing: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, stream: bool = Query(False)):
    """Chat endpoint using RAG for enhanced responses"""
    try:
        # Get the RAG system
        rag_system = get_rag_system()
        
        # Handle streaming response
        if stream:
            async def event_generator():
                try:
                    response_stream = rag_system.query(
                        request.question,
                        request.profile,
                        request.ranking,
                        stream=True
                    )
                    
                    # Stream each token as a separate SSE event
                    for token in response_stream:
                        if token:
                            # Format properly for SSE - data field only
                            yield {"data": token.strip()}
                            # Small delay to ensure frontend can process tokens
                            await asyncio.sleep(0.01)
                    
                    # Send a completion event to signal the end of the stream
                    yield {"event": "done", "data": ""}
                    
                except Exception as e:
                    print(f"Error in stream: {str(e)}")
                    yield {"data": f"Error: {str(e)}"}
                    yield {"event": "done", "data": ""}
            
            # Ensure proper SSE headers and CORS compatibility
            return EventSourceResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Important for Nginx
                    "Access-Control-Allow-Origin": "*",
                }
            )
        
        # Regular non-streaming response
        response = rag_system.query(
            request.question,
            request.profile,
            request.ranking,
            stream=False
        )
        
        return ChatResponse(
            answer=response["answer"],
            citations=response["citations"]
        )
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/nvidia-rag", response_model=NvidiaRagResponse)
async def nvidia_rag_query(request: NvidiaRagRequest):
    """Endpoint for NVIDIA API-powered RAG queries"""
    global nvidia_vector_store, nvidia_llm
    
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=501, detail="NVIDIA RAG not available: API key not set")
    
    if nvidia_vector_store is None:
        raise HTTPException(status_code=501, detail="NVIDIA RAG not available: Vector store not initialized")
    
    # Use the requested model if different from current
    if request.model != nvidia_llm.model:
        try:
            nvidia_llm = ChatNVIDIA(
                model=request.model,
                api_key=NVIDIA_API_KEY
            )
        except Exception as e:
            print(f"Error switching model: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error switching to model {request.model}: {str(e)}")
    
    try:
        # Create retriever
        retriever = nvidia_vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Format framework ranking for prompt
        framework_info = ""
        if request.framework_ranking:
            framework_info = "Model Predictions (frameworks ranked by suitability for your requirements):\n"
            
            # Only show top 5 frameworks
            top_frameworks = request.framework_ranking[:5]
            
            for i, rank in enumerate(top_frameworks):
                percentage = rank.get("percentage", 0)
                if not percentage and "prob" in rank:
                    # Convert probability to percentage if needed
                    prob = rank["prob"]
                    percentage = prob * 100 if prob <= 1 else prob
                
                framework = rank.get('framework', 'Unknown')
                
                # Add descriptive text based on ranking
                description = ""
                if i == 0:
                    description = " (best match)"
                elif i == 1:
                    description = " (strong alternative)"
                elif i == 2:
                    description = " (viable option)"
                
                framework_info += f"- {framework}{description}: {percentage:.1f}%\n"
        
        # Format user profile for prompt
        profile_info = ""
        if request.user_profile:
            profile_info = "User Hardware and Requirements:\n"
            
            # Hardware info
            hw_info = []
            if request.user_profile.get("hw_cpu", 0) == 1:
                hw_info.append("CPU")
            if request.user_profile.get("hw_nvidia", 0) == 1:
                hw_info.append("NVIDIA GPU")
            if request.user_profile.get("hw_amd", 0) == 1:
                hw_info.append("AMD GPU")
            if request.user_profile.get("hw_other", 0) == 1:
                hw_info.append("Other accelerators")
            
            profile_info += f"- Hardware: {', '.join(hw_info) if hw_info else 'Not specified'}\n"
            
            # Cross-vendor support
            if request.user_profile.get("need_cross_vendor", 0) == 1:
                profile_info += "- Needs cross-vendor support\n"
            
            # Weights using descriptive terms
            def weight_to_text(weight):
                if weight < 0.3:
                    return "Low importance"
                elif weight < 0.7:
                    return "Medium importance" 
                else:
                    return "High importance"
            
            # Weights
            perf_weight = request.user_profile.get("perf_weight", 0.5)
            port_weight = request.user_profile.get("port_weight", 0.5)
            eco_weight = request.user_profile.get("eco_weight", 0.5)
            
            profile_info += f"- Performance: {weight_to_text(perf_weight)}\n"
            profile_info += f"- Portability across devices: {weight_to_text(port_weight)}\n"
            profile_info += f"- Ecosystem maturity: {weight_to_text(eco_weight)}\n"
            
            # Project type
            project_type = ""
            if request.user_profile.get("greenfield", 0) == 1:
                project_type = "New project (starting from scratch)"
            elif request.user_profile.get("gpu_extend", 0) == 1:
                project_type = "Extending existing GPU code"
            elif request.user_profile.get("cpu_port", 0) == 1:
                project_type = "Porting from CPU code to GPU"
            
            if project_type:
                profile_info += f"- Project type: {project_type}\n"
            
            # Domain with more description
            domains = []
            domain_mapping = {
                "domain_ai_ml": "AI/ML",
                "domain_hpc": "High Performance Computing",
                "domain_climate": "Climate/Weather Simulation",
                "domain_embedded": "Embedded Systems",
                "domain_graphics": "Graphics/Visualization",
                "domain_data_analytics": "Data Analytics",
                "domain_other": "Other domain"
            }
            
            for key, label in domain_mapping.items():
                if request.user_profile.get(key, 0) == 1:
                    domains.append(label)
            
            if domains:
                profile_info += f"- Application domain: {', '.join(domains)}\n"
            
            # Additional preferences with clearer descriptions
            if request.user_profile.get("pref_directives", 0) == 1:
                profile_info += "- Preferred coding style: Directive-based (like OpenMP/OpenACC)\n"
            if request.user_profile.get("pref_kernels", 0) == 1:
                profile_info += "- Preferred coding style: Explicit kernel-based (like CUDA/HIP)\n"
            
            # Skill level as descriptive text
            skill_level = request.user_profile.get("gpu_skill_level", 1)
            skill_text = "No prior experience with GPU programming"
            if skill_level == 1:
                skill_text = "Basic understanding of GPU programming"
            elif skill_level == 2:
                skill_text = "Intermediate GPU programming skills"
            elif skill_level == 3:
                skill_text = "Expert-level GPU programming knowledge"
            
            profile_info += f"- Experience level: {skill_text}\n"
            
            # Vendor lock-in tolerance in clearer terms
            lockin = request.user_profile.get("lockin_tolerance", 0.5)
            if lockin < 0.3:
                lockin_text = "Strong preference for vendor-independent solutions"
            elif lockin < 0.7:
                lockin_text = "Moderate concern about vendor lock-in"
            else:
                lockin_text = "Comfortable with vendor-specific solutions"
            
            profile_info += f"- Vendor lock-in preference: {lockin_text}\n"
        
        # Define RAG prompt with user context
        template = """You are an AI assistant for answering questions about HPC (High Performance Computing) frameworks and programming models.
        You provide accurate, helpful responses tailored to the user's specific hardware and requirements.
        
        FRAMEWORK RANKING:
        {framework_info}
        
        USER REQUIREMENTS:
        {profile_info}
        
        RESEARCH CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        Provide a focused, technical response that directly addresses the question. Use the information above to:
        1. Explain WHY specific frameworks are suitable based on the user's hardware and requirements
        2. Mention relevant technical features from the research context
        3. Compare frameworks when appropriate for the question
        4. Address trade-offs in performance, portability, and ecosystem
        5. Use direct address (you/your) rather than third-person references
        
        Be concise and technical, avoiding repetition of ranking percentages. If you don't know something specific, say so."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        rag_chain = (
            {
                "context": retriever, 
                "question": RunnablePassthrough(),
                "framework_info": lambda _: framework_info,
                "profile_info": lambda _: profile_info
            }
            | prompt
            | nvidia_llm
            | StrOutputParser()
        )
        
        # Print the full prompt that will be sent to the LLM
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n--- PROMPT SENT TO LLM [{timestamp}] ---")
            print(f"Query: '{request.query}'")
            print("-" * 40)
            
            # Get the actual documents that will be included in the context
            retrieved_docs = retriever.invoke(request.query)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Format the actual prompt that will be sent to the LLM
            full_prompt = template.format(
                framework_info=framework_info,
                profile_info=profile_info,
                context=context_text,
                question=request.query
            )
            
            # Print ONLY what the model actually receives
            print(full_prompt)
            print("--- END OF PROMPT ---\n")
        except Exception as e:
            print(f"Error printing full prompt: {str(e)}")
        
        # Process query
        response = rag_chain.invoke(request.query)
        
        return NvidiaRagResponse(response=response)
    except Exception as e:
        print(f"Error in NVIDIA RAG query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"NVIDIA RAG query error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 