import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, List


def predict_and_evaluate(model, X_test: np.ndarray, y_test: np.ndarray, class_labels: List[str]) -> Dict[str, Any]:
    """
    Predicts and evaluates the model, returning a dictionary of results.
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Extract key metrics
    results = {
        'Accuracy': accuracy,
        'Precision (<=50K)': report['0']['precision'],
        'Recall (<=50K)': report['0']['recall'],
        'F1-Score (<=50K)': report['0']['f1-score'],
        'Precision (>50K)': report['1']['precision'],
        'Recall (>50K)': report['1']['recall'],
        'F1-Score (>50K)': report['1']['f1-score'],
        'Prediction Time (s)': end_time - start_time,
        'Confusion Matrix': conf_matrix.tolist()
    }

    # Print a nice summary
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    return results
