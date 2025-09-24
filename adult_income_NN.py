import numpy as np
from sklearn.neural_network import MLPClassifier
import time
from elemetns.load_data import DataPipeline
from elemetns.predict import predict_and_evaluate
from elemetns.save_results import save_results
from elemetns.visualize import visualize_confusion_matrix


# """Main execution of the Adult Income prediction with a Neural Network."""
# 1. Load and preprocess the data using the DataPipeline
print("--- Loading and Preprocessing Data ---")
data_pipeline = DataPipeline(dataset_path='adult.csv')
(train_x, train_y), (val_x, val_y), (test_x, test_y) = data_pipeline.load_dataset()
class_labels = data_pipeline.classes

# 2. Define the Neural Network Model
# We'll use a simple Multi-layer Perceptron (MLP) for this example
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # A simple two-layer network
    max_iter=500,
    random_state=42,
    activation='relu',
    solver='adam',
    verbose=True
)

# 3. Train the Model
print("\n--- Training the Neural Network ---")
start_train_time = time.time()
model.fit(train_x, train_y)
end_train_time = time.time()
print(f"Training Time: {end_train_time - start_train_time:.2f} seconds")

# 4. Predict and Evaluate on Test Set
print("\n--- Evaluating the Model on the Test Set ---")
results = predict_and_evaluate(model, test_x, test_y, class_labels)

# 5. Visualize Results
print("\n--- Visualizing Confusion Matrix ---")
visualize_confusion_matrix(np.array(results['Confusion Matrix']), class_labels)

# 6. Save Results
print("\n--- Saving Results ---")
save_results('Neural Network Classifier', results)


