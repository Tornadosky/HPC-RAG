import streamlit as st
import requests
import json
from typing import List, Dict, Any
import os
import time

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Set page configuration
st.set_page_config(
    page_title="HPC Framework Recommender with RAG",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e9ecef;
    }
    .chat-message.assistant {
        background-color: #d2f8d2;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 10px;
    }
    .chat-message .content {
        width: 100%;
    }
    .sources-container {
        margin-top: 10px;
        padding: 10px;
        background-color: #f1f3f5;
        border-radius: 0.5rem;
    }
    .sources-toggle {
        cursor: pointer;
        color: #0066cc;
        font-weight: bold;
    }
    .sources-content {
        padding: 10px 0;
    }
    .stProgress .st-bo {
        background-color: #4CAF50 !important;
    }
    .framework-bar {
        height: 30px;
        margin-bottom: 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Function to call the /predict endpoint
def get_framework_prediction(user_profile):
    try:
        response = requests.post(f"{API_URL}/predict", json=user_profile)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting prediction: {str(e)}")
        return None

# Function to send chat message to the /chat endpoint
def send_chat_message(question, user_profile, ranking, stream=True):
    try:
        if stream:
            # For streaming, we need to handle the event stream manually
            params = {"stream": "true"}
            headers = {'Content-Type': 'application/json'}
            data = {"question": question, "profile": user_profile, "ranking": ranking}
            
            response = requests.post(
                f"{API_URL}/chat",
                params=params,
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()
            return response
        else:
            # For non-streaming, we can just get the JSON response
            response = requests.post(
                f"{API_URL}/chat",
                json={"question": question, "profile": user_profile, "ranking": ranking}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error sending chat message: {str(e)}")
        return None

# Function to parse SSE stream data
def parse_sse(response):
    answer = ""
    last_token = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data:'):
                data = line[5:].strip()
                if data:
                    # Check if we need to add a space between tokens
                    # Only add a space if the last token doesn't end with whitespace
                    # and the current token doesn't start with whitespace or punctuation
                    if (answer and last_token and 
                        not last_token[-1].isspace() and 
                        not data[0].isspace() and 
                        not data[0] in '.,;:!?)-]}'):
                        answer += " "
                    
                    answer += data
                    last_token = data
                    yield answer

# Initialize session state for chat history if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Title and description
st.title("🚀 HPC Framework Recommender with RAG")
st.markdown("""
This app helps you find the best HPC framework for your needs and provides detailed information using Retrieval-Augmented Generation (RAG) technology.
""")

# Sidebar for user profile inputs
with st.sidebar:
    st.header("Your HPC Profile")
    
    # Hardware options
    st.subheader("Hardware")
    hw_cpu = st.checkbox("CPU", value=True)
    hw_nvidia = st.checkbox("NVIDIA GPU")
    hw_amd = st.checkbox("AMD GPU")
    hw_other = st.checkbox("Other accelerators")
    need_cross_vendor = st.checkbox("Need cross-vendor support")
    
    # Preference weights
    st.subheader("Preference Weights")
    perf_weight = st.slider("Performance", 0.0, 1.0, 0.5)
    port_weight = st.slider("Portability", 0.0, 1.0, 0.5)
    eco_weight = st.slider("Ecosystem", 0.0, 1.0, 0.5)
    
    # Programming model preference
    st.subheader("Programming Model")
    pref_directives = st.checkbox("Prefer directive-based")
    pref_kernels = st.checkbox("Prefer kernel-based")
    
    # Project type
    st.subheader("Project Type")
    project_type = st.radio(
        "Select project type",
        ["New project", "Extending GPU code", "Porting from CPU"],
        index=0
    )
    
    # Domain
    st.subheader("Domain")
    domain = st.selectbox(
        "Select domain",
        ["AI/ML", "HPC", "Climate", "Embedded", "Graphics", "Data Analytics", "Other"],
        index=1
    )
    
    # Other preferences
    st.subheader("Other Preferences")
    lockin_tolerance = st.slider("Vendor lock-in tolerance", 0.0, 1.0, 0.5)
    gpu_skill_level = st.slider("GPU programming skill level", 1, 5, 2)
    
    # Submit button
    if st.button("Get Recommendations"):
        # Create user profile from inputs
        user_profile = {
            "hw_cpu": 1 if hw_cpu else 0,
            "hw_nvidia": 1 if hw_nvidia else 0,
            "hw_amd": 1 if hw_amd else 0,
            "hw_other": 1 if hw_other else 0,
            "need_cross_vendor": 1 if need_cross_vendor else 0,
            "perf_weight": perf_weight,
            "port_weight": port_weight,
            "eco_weight": eco_weight,
            "pref_directives": 1 if pref_directives else 0,
            "pref_kernels": 1 if pref_kernels else 0,
            "greenfield": 1 if project_type == "New project" else 0,
            "gpu_extend": 1 if project_type == "Extending GPU code" else 0,
            "cpu_port": 1 if project_type == "Porting from CPU" else 0,
            "domain_ai_ml": 1 if domain == "AI/ML" else 0,
            "domain_hpc": 1 if domain == "HPC" else 0,
            "domain_climate": 1 if domain == "Climate" else 0,
            "domain_embedded": 1 if domain == "Embedded" else 0,
            "domain_graphics": 1 if domain == "Graphics" else 0,
            "domain_data_analytics": 1 if domain == "Data Analytics" else 0,
            "domain_other": 1 if domain == "Other" else 0,
            "lockin_tolerance": lockin_tolerance,
            "gpu_skill_level": gpu_skill_level
        }
        
        st.session_state.user_profile = user_profile
        with st.spinner("Getting framework predictions..."):
            st.session_state.prediction = get_framework_prediction(user_profile)
        
        # Clear chat history when getting new recommendations
        st.session_state.messages = []

# Main content area
col1, col2 = st.columns([1, 1])

# Display the prediction results in the first column
with col1:
    st.header("Framework Recommendations")
    
    if st.session_state.prediction:
        prediction = st.session_state.prediction
        
        # Display explanation
        st.markdown(f"**Recommendation:** {prediction['explanation']}")
        
        # Display bar chart for top 5 frameworks
        st.subheader("Top Frameworks")
        
        # Sort and take top 5
        sorted_ranking = sorted(prediction["ranking"], key=lambda x: x["prob"], reverse=True)[:5]
        
        # Calculate max probability for scaling
        max_prob = sorted_ranking[0]["prob"]
        
        # Display bars
        for rank in sorted_ranking:
            framework = rank["framework"]
            prob = rank["prob"]
            percentage = int(prob * 100)
            
            # Create a progress bar visualization
            st.markdown(f"**{framework}**")
            st.progress(prob)
            st.markdown(f"{percentage}%")
            st.markdown("---")
    else:
        st.info("Please fill out your profile and click 'Get Recommendations' to see framework suggestions.")

# Chat interface in the second column
with col2:
    st.header("HPC Framework Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations if they exist
            if message.get("citations"):
                with st.expander("Sources"):
                    for citation in message["citations"]:
                        st.markdown(f"- {citation}")
    
    # Input for new question
    if prompt := st.chat_input("Ask a question about HPC frameworks..."):
        if not st.session_state.prediction:
            st.error("Please get recommendations first before asking questions.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response from RAG system
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # For storing citations
                citations_container = st.container()
                
                # Get ranking from prediction
                ranking = st.session_state.prediction["ranking"]
                
                with st.spinner("Thinking..."):
                    # Handle streaming response
                    response = send_chat_message(prompt, st.session_state.user_profile, ranking, stream=True)
                    
                    full_response = ""
                    for partial_response in parse_sse(response):
                        full_response = partial_response
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)
                        
                    # Final response without cursor
                    message_placeholder.markdown(full_response)
                    
                    # Extract citations using a separate API call to get the complete response with citations
                    complete_response = send_chat_message(prompt, st.session_state.user_profile, ranking, stream=False)
                    
                    if complete_response and "citations" in complete_response:
                        citations = complete_response["citations"]
                        
                        # Display citations in an expander
                        if citations:
                            with citations_container.expander("Sources"):
                                for citation in citations:
                                    st.markdown(f"- {citation}")
                                    
                            # Store in session for history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": full_response,
                                "citations": citations
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": full_response
                            })
                    else:
                        # If we couldn't get citations, just store the response
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })

# Footer
st.markdown("---")
st.markdown("HPC Framework Recommender with RAG - Built with Streamlit, FastAPI, and LangChain")

if __name__ == "__main__":
    # This will be executed when the script is run directly
    # The Streamlit app is already defined above
    pass 