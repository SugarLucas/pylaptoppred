import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def preprocess_data(df):
    """
    Preprocesses the data by removing non-numeric columns.

    Args:
        df: The pandas DataFrame containing the dataset.

    Returns:
        df: The preprocessed DataFrame with only numeric columns.
    """
    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    return df_numeric

def split_data(df, target, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets after preprocessing.

    Args:
        df: The pandas DataFrame containing the dataset.
        target: The name of the target variable column.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
        X_train, X_test, y_train, y_test: The split data as pandas DataFrames.
    """
    df = preprocess_data(df)  # Preprocess data to keep only numeric columns
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Trains a LogisticRegression model on the training set with an increased max_iter parameter.

    Args:
        X_train: The training input samples.
        y_train: The target values for the training input samples.

    Returns:
        The trained model.
    """
    model = LogisticRegression(max_iter=1000)  # Instantiate LogisticRegression with increased max_iter
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the testing set.

    Args:
        model: The trained machine learning model.
        X_test: The testing input samples.
        y_test: The true values for the testing input samples.

    Returns:
        A dictionary containing accuracy, precision, and recall.
    """
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='macro'),
        "recall": recall_score(y_test, predictions, average='macro')
    }
    return metrics

def plot_metrics(metrics, filename, directory):
    """
    Generates and saves a bar plot of the evaluation metrics.

    Args:
        metrics: A dictionary containing the evaluation metrics.
        filename: The filename for the saved plot.
        directory: The directory to save the plot in.

    Raises:
        ValueError: If no filename is provided.
        OSError: If the specified directory does not exist.
    """
    if filename == "":
        raise ValueError("No filename provided.")
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title('Model Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, filename))
    plt.close()
