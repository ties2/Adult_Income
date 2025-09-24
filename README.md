# Adult Income Prediction with a Neural Network

## Project Overview
This project predicts whether an individual's income exceeds **$50,000 annually** using the **1994 U.S. Census dataset**.  
It demonstrates a complete machine learning workflow — from **data preprocessing** to **building and evaluating** a neural network model.

---

## Dataset
- **Source:** [Adult Income Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income)  
- **Features:** Includes attributes such as **age, education, occupation, capital gain**, etc.  
- **Target:** Classifying individuals into:
  - `<=50K`
  - `>50K`

---

## Methodology
1. **Data Preprocessing**  
   - Handled missing values (`?`).  
   - Encoded categorical features into numeric form.  
   - Dropped irrelevant column: `fnlwgt`.

2. **Feature Scaling**  
   - Applied **StandardScaler** on numerical features to improve neural network convergence.

3. **Model Training**  
   - Implemented a **Multi-layer Perceptron (MLPClassifier)** from scikit-learn.  

4. **Model Evaluation**  
   - Metrics: **Accuracy, Precision, Recall, F1-score**  
   - Visualized performance with a **Confusion Matrix**.

---

## Results
- The trained neural network achieved **high accuracy** on the test set.  
- The **confusion matrix** showed reliable classification of both income categories.  

---

## Future Improvements
- Experiment with advanced architectures using **TensorFlow** or **PyTorch**.  
- Perform **hyperparameter tuning** to optimize MLP performance.  
- Apply **feature engineering** to create new informative variables.  
- Compare performance with other models like **Random Forest** and **Gradient Boosting Machines**.  

---
