import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
data = pd.read_csv('data_aug.csv')

# Display basic information about the dataset
print(f"Dataset shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())
print("\nFramework distribution:")
print(data['framework'].value_counts())

# Preprocess the data
X = data.drop(['user_id', 'framework'], axis=1)
y = data['framework']

# Convert categorical data to numerical if needed
# In this case, all features are already numerical

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Feature scaling may not be necessary for tree-based models, but included for completeness
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an interpretable model - Random Forest with calibrated probabilities
print("\nTraining Random Forest model...")
base_rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5,  # Limit depth for interpretability
    min_samples_leaf=5,
    random_state=42
)

# Use CalibratedClassifierCV for balanced probabilities
# The calibration helps ensure probabilities are well-calibrated
calibrated_rf = CalibratedClassifierCV(base_rf, method='sigmoid', cv=5)
calibrated_rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = calibrated_rf.predict(X_test_scaled)
y_pred_proba = calibrated_rf.predict_proba(X_test_scaled)

# Evaluate the model
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=calibrated_rf.classes_, 
            yticklabels=calibrated_rf.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Feature importance for interpretability
print("\nFeature Importance:")

# Calculate permutation importance (more reliable than default importance)
perm_importance = permutation_importance(calibrated_rf, X_test_scaled, y_test, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Sample predictions with probabilities
print("\nSample predictions with probabilities:")
sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = calibrated_rf.predict(X_test_scaled[idx].reshape(1, -1))[0]
    probs = calibrated_rf.predict_proba(X_test_scaled[idx].reshape(1, -1))[0]
    
    print(f"\nSample {idx}:")
    print(f"Actual framework: {actual}")
    print(f"Predicted framework: {predicted}")
    print("Probabilities:")
    for i, (cls, prob) in enumerate(zip(calibrated_rf.classes_, probs)):
        print(f"  {cls}: {prob:.4f}")

# Train a decision tree for visualization (more interpretable)
print("\nTraining a Decision Tree for visualization...")
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train_scaled, y_train)

# Decision tree visualization
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=X.columns.tolist(), class_names=dt.classes_.tolist(), rounded=True)
plt.title("Decision Tree Visualization")
plt.savefig('decision_tree.png', bbox_inches='tight')

# Save the model and scaler for later use
joblib.dump((calibrated_rf, scaler), 'model.joblib')
print("\nModel saved as 'model.joblib'")

print("\nTraining and evaluation complete. Visualizations saved as PNG files.")

# Function to make predictions on new data
def predict_framework(new_data):
    """
    Make predictions on new data.
    
    Parameters:
    -----------
    new_data : pandas DataFrame or numpy array
        New data to make predictions on.
        
    Returns:
    --------
    predictions : list
        Predicted framework.
    probabilities : array
        Probability for each class.
    classes : array
        Class names corresponding to the probabilities.
    """
    if isinstance(new_data, pd.DataFrame):
        # Drop user_id and framework if they exist
        cols_to_drop = [col for col in ['user_id', 'framework'] if col in new_data.columns]
        new_data = new_data.drop(cols_to_drop, axis=1, errors='ignore')
        
    # Scale the data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = calibrated_rf.predict(new_data_scaled)
    probabilities = calibrated_rf.predict_proba(new_data_scaled)
    
    return predictions, probabilities, calibrated_rf.classes_

# Example usage:
# predictions, probabilities, classes = predict_framework(some_new_data) 