import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Define paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_dir, "datasets", "CICIDS2017_hf", "random", "train-00000-of-00001.parquet")
    results_dir = os.path.join(project_dir, "results", "baseline")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading dataset from: {dataset_path}")
    # Load just 100,000 rows for baseline
    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        print(f"Error loading parquet: {e}. Make sure git lfs pull was successful.")
        return

    print(f"Original shape: {df.shape}")
    
    # Sample 100k rows for fast baseline
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)
    
    print(f"Sampled shape: {df.shape}")

    # Prepare features and labels
    # 'label' is binary (0=BENIGN, 1=Attack), 'Label' is multiclass
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
        
    X = df.drop(columns=['label'])
    y = df['label']

    # Handle any remaining NaN/Inf values just in case
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Train model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict
    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics_df = pd.DataFrame([{
        "Model": "RandomForest",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "Train_Samples": len(X_train),
        "Test_Samples": len(X_test)
    }])
    
    metrics_path = os.path.join(results_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    print(metrics_df)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'Attack'], yticklabels=['BENIGN', 'Attack'])
    plt.title('Confusion Matrix - Baseline')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    print("Baseline execution completed successfully.")

if __name__ == "__main__":
    main()
