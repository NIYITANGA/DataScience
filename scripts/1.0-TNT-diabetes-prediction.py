"""
Diabetes Prediction Model
Author: TNT
Version: 1.0
Description: Simple machine learning model to predict diabetes using the Pima Indians Diabetes dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load the diabetes dataset"""
    print("Loading diabetes dataset...")
    data = pd.read_csv('data/diabetes.csv')
    print(f"Dataset shape: {data.shape}")
    print(f"Dataset info:")
    print(data.info())
    print(f"\nDataset description:")
    print(data.describe())
    return data

def preprocess_data(data):
    """Preprocess the data"""
    print("\nPreprocessing data...")
    
    # Check for missing values
    print(f"Missing values:\n{data.isnull().sum()}")
    
    # Handle zero values that might represent missing data
    # In this dataset, zeros in certain columns are likely missing values
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_columns:
        if col in data.columns:
            zero_count = (data[col] == 0).sum()
            print(f"Zero values in {col}: {zero_count}")
            # Replace zeros with median for these columns
            if zero_count > 0:
                median_val = data[data[col] != 0][col].median()
                data[col] = data[col].replace(0, median_val)
    
    # Separate features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    print("\nTraining models...")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        trained_models[name] = model
        
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    return results, trained_models, scaler

def save_results(results, trained_models, scaler, feature_names):
    """Save models and results"""
    print("\nSaving results...")
    
    # Create output directories if they don't exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save models
    for name, model in trained_models.items():
        model_filename = f"output/1.0-TNT-{name.lower().replace(' ', '_')}_model_{timestamp}.pkl"
        joblib.dump(model, model_filename)
        print(f"Saved {name} model to {model_filename}")
    
    # Save scaler
    scaler_filename = f"output/1.0-TNT-scaler_{timestamp}.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"Saved scaler to {scaler_filename}")
    
    # Save results to text file
    results_filename = f"results/1.0-TNT-model_results_{timestamp}.txt"
    with open(results_filename, 'w') as f:
        f.write("Diabetes Prediction Model Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for name, result in results.items():
            f.write(f"\n{name} Results:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"ROC-AUC: {result['roc_auc']:.4f}\n")
            f.write(f"\nClassification Report:\n{result['classification_report']}\n")
            f.write(f"\nConfusion Matrix:\n{result['confusion_matrix']}\n")
    
    print(f"Saved results to {results_filename}")
    
    # Create and save visualizations
    create_visualizations(results, timestamp)
    
    return timestamp

def create_visualizations(results, timestamp):
    """Create and save visualizations"""
    print("Creating visualizations...")
    
    # Model comparison plot
    plt.figure(figsize=(12, 5))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # ROC-AUC comparison
    plt.subplot(1, 2, 2)
    roc_aucs = [results[model]['roc_auc'] for model in models]
    plt.bar(models, roc_aucs, color=['lightgreen', 'orange'])
    plt.title('Model ROC-AUC Comparison')
    plt.ylabel('ROC-AUC')
    plt.ylim(0, 1)
    for i, v in enumerate(roc_aucs):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'results/1.0-TNT-model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (name, result) in enumerate(results.items()):
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} - Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'results/1.0-TNT-confusion_matrices_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualizations to results/ directory")

def main():
    """Main function to run the diabetes prediction pipeline"""
    print("Starting Diabetes Prediction Model Training...")
    print("=" * 50)
    
    # Load data
    data = load_data()
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    results, trained_models, scaler = train_models(X_train, X_test, y_train, y_test)
    
    # Save results
    timestamp = save_results(results, trained_models, scaler, X.columns.tolist())
    
    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 50)
    
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"Best performing model: {best_model}")
    print(f"Best accuracy: {results[best_model]['accuracy']:.4f}")
    print(f"Best ROC-AUC: {results[best_model]['roc_auc']:.4f}")
    
    print(f"\nFiles saved with timestamp: {timestamp}")
    print("- Models saved in 'output/' directory")
    print("- Results and visualizations saved in 'results/' directory")
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()
