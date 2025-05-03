# HPC-RAG
RAG based solution for choosing HPC frameworks

## Framework Predictor

This project trains a machine learning model to predict HPC frameworks based on user preferences and hardware configurations.

### Features

- Interpretable model training using Random Forest and Decision Tree
- Calibrated probabilities for balanced predictions
- Visualization of feature importance and decision tree
- Easy-to-use prediction interface
- Interactive command-line interface for framework recommendations

### Dataset Features

The dataset contains the following features:

**Hardware Information**:
- `hw_cpu`: Has CPU (1 = yes, 0 = no)
- `hw_nvidia`: Has NVIDIA GPU (1 = yes, 0 = no)
- `hw_amd`: Has AMD GPU (1 = yes, 0 = no)
- `hw_other`: Has other accelerators (1 = yes, 0 = no)
- `need_cross_vendor`: Needs cross-vendor support (1 = yes, 0 = no)

**User Preferences**:
- `perf_weight`: Weight given to performance (0.0-1.0)
- `port_weight`: Weight given to portability (0.0-1.0)
- `eco_weight`: Weight given to ecosystem (0.0-1.0)
- `pref_directives`: Prefers directive-based programming (1 = yes, 0 = no)
- `pref_kernels`: Prefers kernel-based programming (1 = yes, 0 = no)
- `lockin_tolerance`: Tolerance for vendor lock-in (0.0-1.0)
- `gpu_skill_level`: GPU programming skill level (1-3, where 3 is highest)

**Project Characteristics**:
- `greenfield`: New project (1 = yes, 0 = no)
- `gpu_extend`: Extending existing GPU code (1 = yes, 0 = no)
- `cpu_port`: Porting from CPU code (1 = yes, 0 = no)

**Application Domain**:
- `domain_ai_ml`: AI/ML domain (1 = yes, 0 = no)
- `domain_hpc`: HPC domain (1 = yes, 0 = no)
- `domain_climate`: Climate domain (1 = yes, 0 = no)
- `domain_embedded`: Embedded domain (1 = yes, 0 = no)
- `domain_graphics`: Graphics domain (1 = yes, 0 = no)
- `domain_data_analytics`: Data analytics domain (1 = yes, 0 = no)
- `domain_other`: Other domain (1 = yes, 0 = no)

**Target (to predict)**:
- `framework`: The recommended HPC framework (CUDA, HIP, OpenCL, SYCL, RAJA, Kokkos, OpenACC, OpenMP)

### Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   ```
   python train_model.py
   ```

3. The script will:
   - Train both a calibrated Random Forest and a Decision Tree model
   - Output performance metrics
   - Generate visualizations in PNG format:
     - `confusion_matrix.png`: Shows prediction accuracy
     - `feature_importance.png`: Shows which features are most important
     - `decision_tree.png`: Visual representation of the decision-making process
   - Save the trained model as `model.joblib`

### Using the Model for Predictions

You can use the model in three ways:

#### 1. Interactive Command-Line Interface

The easiest way to get recommendations is to use the interactive script:

```
python interactive.py
```

This will guide you through a series of questions about your hardware, preferences, and project requirements, then provide a ranked list of framework recommendations with explanations.

#### 2. Using the prediction script with CSV input

```
python predict.py [input_csv_file]
```

If no file is provided, it will use an example input.

A sample input file `sample_input.csv` is provided with example data.

#### 3. Import and use in your own code

```python
import pandas as pd
import joblib

# Load the trained model
calibrated_rf, scaler = joblib.load('model.joblib')

# Create new data with the same features as training data
new_data = pd.DataFrame({
    'hw_cpu': [1],
    'hw_nvidia': [1],
    'hw_amd': [0],
    'hw_other': [0],
    'need_cross_vendor': [0],
    'perf_weight': [0.95],
    'port_weight': [0.20],
    'eco_weight': [0.92],
    'pref_directives': [0],
    'pref_kernels': [1],
    'greenfield': [1],
    'gpu_extend': [0],
    'cpu_port': [0],
    'domain_ai_ml': [1],
    'domain_hpc': [0],
    'domain_climate': [0],
    'domain_embedded': [0],
    'domain_graphics': [0],
    'domain_data_analytics': [0],
    'domain_other': [0],
    'lockin_tolerance': [0.90],
    'gpu_skill_level': [3]
})

# Scale the data
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = calibrated_rf.predict(new_data_scaled)
probabilities = calibrated_rf.predict_proba(new_data_scaled)

# Print results
print(f"Predicted framework: {predictions[0]}")
print("Probabilities:")
# Sort probabilities in descending order for better readability
sorted_probs = sorted(zip(calibrated_rf.classes_, probabilities[0]), key=lambda x: x[1], reverse=True)
for cls, prob in sorted_probs:
    print(f"  {cls}: {prob:.4f} ({prob*100:.2f}%)")
```
