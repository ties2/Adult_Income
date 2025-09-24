import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def visualize_confusion_matrix(conf_matrix: np.ndarray, class_labels: List[str]):
    """
    Visualizes the confusion matrix using a heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
