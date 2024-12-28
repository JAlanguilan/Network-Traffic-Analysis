from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.model_selection import train_test_split


def direct_multiclass_train(model_name, X_train, y_train, **kwargs):
    """
    Trains a multiclass classification model.

    Args:
        model_name (str): Model type ('mlp' or 'rf').
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        **kwargs: Additional hyperparameters for the model.

    Returns:
        model: Trained model.
    """

    # Define the available models
    models = {
        'mlp': MLPClassifier(random_state=42, **kwargs),
        'rf': RandomForestClassifier(random_state=42, **kwargs),
    }

    # Check if the provided model name is valid
    if model_name not in models:
        raise ValueError("model_name should be 'mlp' or 'rf'")
    
    # Select the model based on the provided model name
    model = models[model_name]
    model.fit(X_train, y_train)
    return model

def direct_multiclass_test(model, X_test, y_test, average='weighted'):
    """
    Tests a trained model and calculates evaluation metrics.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.
        average (str): Averaging method for precision, recall, and F1.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    
    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=average),
        'recall': recall_score(y_test, y_pred, average=average),
        'f1_score': f1_score(y_test, y_pred, average=average),
    }
    return metrics

def data_resampling(df, target_col, sampling_strategy):
    """
    Performs random undersampling to balance the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target column.
        sampling_strategy (float/str): Undersampling strategy.

    Returns:
        pd.DataFrame: Resampled DataFrame.
    """

    # Check if the target column exists in the DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Separate features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Check if there is more than one class in the target column
    if len(y.unique()) < 2:
        print("Warning: Only one class found, skipping resampling")
        return df
    
    # Perform random undersampling to balance the dataset
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # Combine the resampled features and target into a single DataFrame
    return pd.concat([X_resampled, y_resampled], axis=1)

def improved_data_split(df, target_col, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets with stratification.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column for stratification.
        test_size (float): Proportion of test data.
        random_state (int): Random seed.

    Returns:
        tuple: Training and testing DataFrames.
    """

    # Check if the target column exists in the DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Split the data into training and testing sets with stratification
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )

def get_binary_dataset(df, target_col, positive_class='Benign', new_label=('Benign', 'Malicious')):
    """
    Converts a multiclass dataset into a binary dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column name.
        positive_class (str): Class to be treated as positive.
        new_label (tuple): Tuple for renaming (positive_label, negative_label).

    Returns:
        pd.DataFrame: Binary-labeled DataFrame.
    """

    # Create a copy of the DataFrame to avoid modifying the original data
    df_binary = df.copy()

    # Apply a function to the target column to convert it to binary labels
    df_binary[target_col] = df_binary[target_col].apply(
        lambda x: new_label[0] if x == positive_class else new_label[1])
    
    return df_binary