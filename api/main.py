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

# Import RAG functionality
from rag import get_rag_system

app = FastAPI(title="HPC Framework Recommender API", 
             description="API for recommending HPC frameworks based on user preferences")

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
        explanation_parts.append(f"Your hardware ({', '.join(hardware_used)})")
    
    if preferences:
        explanation_parts.append(f"emphasis on {', '.join(preferences)}")
    
    if prog_model:
        explanation_parts.append(f"preference for {', '.join(prog_model)}")
    
    if project_type:
        explanation_parts.append(f"{project_type}")
    
    if input_data["need_cross_vendor"] == 1:
        explanation_parts.append("need for cross-vendor support")
    
    if input_data["lockin_tolerance"] > 0.8:
        explanation_parts.append("high vendor lock-in tolerance")
    elif input_data["lockin_tolerance"] < 0.3:
        explanation_parts.append("low vendor lock-in tolerance")
    
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
                    
                    for token in response_stream:
                        if token:
                            yield {"data": token}
                except Exception as e:
                    print(f"Error in stream: {str(e)}")
                    yield {"data": f"Error: {str(e)}"}
            
            return EventSourceResponse(event_generator())
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 