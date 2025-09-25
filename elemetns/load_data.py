import numpy as np
import pandas as pd
from typing import Type, List, Any, Iterable, Union, Set, Optional, Callable, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


class DataPipeline:
    def __init__(self, dataset_path: str = './adult.csv', target_column: str = 'income'):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.classes = None
        self.scaler = None
        self.pca = None

    def load_dataset(self,
                     test_size: float = 0.2,
                     val_size: float = 0.25,
                     normalize: bool = True,
                     scaling_method: str = 'standard',
                     apply_pca: bool = False,
                     pca_components: Optional[int] = None,
                     subsample: Optional[int] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        """
        Args:
            test_size: Proportion of the dataset to include in the test split.
            val_size: Proportion of the remaining data to include in the validation split.
            normalize: Whether to normalize the numerical data.
            scaling_method: 'standard', 'minmax', or 'robust'.
            apply_pca: Whether to apply PCA dimensionality reduction.
            pca_components: Number of PCA components (None for automatic, keeping 95% variance).
            subsample: If specified, subsample the data for faster experimentation.
        Returns:
            A tuple containing the splits: (train_x, train_y), (val_x, val_y), (test_x, test_y).
        """
        print(f"Loading {self.dataset_path} dataset...")
        df = pd.read_csv(self.dataset_path, skipinitialspace=True)

        # Handle missing values ('?') by replacing and dropping rows
        df = df.replace('?', np.nan).dropna()

        # Drop the 'fnlwgt' column as it's not relevant for prediction
        if 'fnlwgt' in df.columns:
            df = df.drop('fnlwgt', axis=1)

        # Subsample if requested
        if subsample:
            df = df.sample(n=subsample, random_state=42)

        # Separate features (X) and target (y)
        y = df[self.target_column].apply(lambda x: 1 if x == '>50K' else 0).values
        X = df.drop(self.target_column, axis=1)

        # Get class names
        self.classes = ['<=50K', '>50K']

        # Identify categorical and numerical features
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

        # One-hot encode categorical features
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

        # Split data into training and a temporary set
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, random_state=42, stratify=y
        )

        # Split temporary set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size / (test_size + val_size), random_state=42, stratify=y_temp
        )

        # Convert to numpy arrays after splitting
        train_x, train_y = X_train.values, y_train
        val_x, val_y = X_val.values, y_val
        test_x, test_y = X_test.values, y_test

        # Get numerical features list after one-hot encoding
        numerical_indices = [X.columns.get_loc(c) for c in numerical_features]

        # Normalize/Scale the data
        if normalize:
            print(f"Applying {scaling_method} scaling...")
            train_x, val_x, test_x = self._normalize_data(
                train_x, val_x, test_x, numerical_indices, method=scaling_method
            )

        # Apply PCA if requested
        if apply_pca:
            print(f"Applying PCA (components={pca_components or 'auto'})...")
            train_x, val_x, test_x = self._apply_pca(
                train_x, val_x, test_x, n_components=pca_components
            )

        # Print dataset info
        self._print_dataset_info(train_x, val_x, test_x)

        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    def _normalize_data(self, train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray,
                        numerical_indices: List[int], method: str = 'standard') -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Normalize the numerical columns of the data using specified method"""

        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Scaling method '{method}' is not yet implemented. Please use 'standard'.")

        train_x[:, numerical_indices] = self.scaler.fit_transform(train_x[:, numerical_indices])
        val_x[:, numerical_indices] = self.scaler.transform(val_x[:, numerical_indices])
        test_x[:, numerical_indices] = self.scaler.transform(test_x[:, numerical_indices])

        return train_x, val_x, test_x

    def _apply_pca(self, train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray,
                   n_components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply PCA dimensionality reduction"""

        if n_components is None:
            self.pca = PCA(n_components=0.95)
        else:
            self.pca = PCA(n_components=n_components)

        train_x = self.pca.fit_transform(train_x)
        val_x = self.pca.transform(val_x)
        test_x = self.pca.transform(test_x)

        print(f"PCA reduced dimensions from {self.pca.n_features_in_} to {self.pca.n_components_}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

        return train_x, val_x, test_x

    def _print_dataset_info(self, train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray):
        """Print dataset information"""
        print("\n" + "=" * 50)
        print("Dataset Information:")
        print("=" * 50)
        print(f"Classes: {self.classes}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Train set: {train_x.shape}")
        print(f"Validation set: {val_x.shape}")
        print(f"Test set: {test_x.shape}")
        print(f"Feature dimension: {train_x.shape[1]}")
        print("=" * 50 + "\n")

    def plot_class_distribution(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """Plot the distribution of classes in each dataset split"""

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, (y, title) in zip(axes, [(y_train, 'Train'), (y_val, 'Validation'), (y_test, 'Test')]):
            unique, counts = np.unique(y, return_counts=True)
            ax.bar(unique, counts)
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(f'{title} Set Class Distribution')
            ax.set_xticks(unique)
            if self.classes:
                ax.set_xticklabels([self.classes[i] for i in unique], rotation=45, ha='right')

        plt.tight_layout()
        plt.show()
