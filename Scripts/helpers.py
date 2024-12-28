import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(fname, target_col=None, visualize=False):
    """
    Loads a CSV file into a DataFrame.
    
    Args:
        fname (str): Path to the CSV file.
        target_col (str, optional): Name of the target column. Defaults to None.
        visualize (bool, optional): Whether to visualize the target column distribution. Defaults to False.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(fname)
    df.columns = df.columns.str.strip()
    print("Columns in DataFrame:", df.columns)
    
    # If target_col is specified and exists in the DataFrame, and visualize is True
    if target_col and target_col in df.columns and visualize:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=target_col, palette='viridis')
        plt.title(f"Distribution of '{target_col}'")
        plt.show()
        
    return df

def clean_data(df, target_col):
    """
    Cleans the DataFrame by:
    - Removing non-numerical columns.
    - Replacing Inf/-Inf with NaN.
    - Filling NaN values with column means.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column to preserve.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Separate target column
    target = df[target_col]
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Replace Inf values and fill NaN with column means
    numerical_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numerical_df = numerical_df.fillna(numerical_df.mean())
    
    # Add target column back
    numerical_df[target_col] = target.values
    return numerical_df

def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column name.
        test_size (float): Proportion of data to include in the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """

    # Check if the target column exists in the DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Separate features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test