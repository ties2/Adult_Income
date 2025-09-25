import os
from typing import Dict, Any


def save_results(classifier_name: str, results_dict: Dict[str, Any]):
    """
    Saves the performance metrics of a classifier to a text file.
    """
    output_dir = './pipeline/output'
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{classifier_name.replace(' ', '_')}_results.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write(f"==================================================\n")
        f.write(f"{classifier_name} Performance:\n")
        f.write(f"==================================================\n")

        for key, value in results_dict.items():
            if key == 'Confusion Matrix':
                f.write(f"{key.replace('_', ' ').title()}:\n")
                f.write(str(value) + '\n')
            elif 'time' in key.lower():
                f.write(f"{key.replace('_', ' ').title()}: {value:.2f}s\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")

    print(f"Results for {classifier_name} saved to {filepath}")
