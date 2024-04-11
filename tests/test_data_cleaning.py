import pandas as pd
import numpy as np
import pytest
import sys
import os
"""
How to run test: pytest tests/test_data_cleaning.py
"""

# Adjust the path to include the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from function_data_cleaning import clean_and_save_data_f


#sample valid/invalid paths
input_path = 'DATA/test.csv'
output_path = 'DATA'
# Sample data creation
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 10, 6, 7, 8, 900],
    'B': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
    'C': [2, 3, 4, 5, 6, 7, 8, 9, 10]
})
df.to_csv(input_path, index=False)


# Ensure the directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)


def test_clean_and_save():
    # Run the cleaning and splitting function
    clean_and_save_data_f(input_path, output_path)
    
    # Read the output files back in
    df_train = pd.read_csv(f"{output_path }_train.csv")
    df_test = pd.read_csv(f"{output_path }_test.csv")

    # Perform your assertions here
    assert not df_train.empty, "Training DataFrame should not be empty"
    assert not df_test.empty, "Test DataFrame should not be empty"
    assert os.path.exists(f"{output_path }_train.csv")
    assert os.path.exists(f"{output_path }_test.csv")

    os.remove(f"{output_path }_train.csv")
    os.remove(f"{output_path }_test.csv")


def test_remove_outliers():
    clean_and_save_data_f(input_path, output_path)

    df_train = pd.read_csv(f"{output_path }_train.csv")
    df_test = pd.read_csv(f"{output_path }_test.csv")
    assert df_train['A'].max() < 900, "Outliers should have been removed from column 'A'"
    assert df_test['A'].max() < 900, "Outliers should have been removed from column 'A'"
    os.remove(f"{output_path }_train.csv")
    os.remove(f"{output_path }_test.csv")

def test_split_proportion():
    clean_and_save_data_f(input_path, output_path)

    df_train = pd.read_csv(f"{output_path }_train.csv")
    df_test = pd.read_csv(f"{output_path }_test.csv")

    total_rows = df_train.shape[0] + df_test.shape[0]
        
    # Then
    assert df_train.shape[0] / total_rows > 0.5, "Training set should be more than 50% of the total data"
    assert df_test.shape[0] / total_rows < 0.5, "Test set should be less than 50% of the total data"
    
    # Cleanup
    os.remove(f"{output_path }_train.csv")
    os.remove(f"{output_path }_test.csv")

def test_missing_data_handle():
    clean_and_save_data_f(input_path, output_path)

    df_train = pd.read_csv(f"{output_path }_train.csv")
    df_test = pd.read_csv(f"{output_path }_test.csv")

    # Then
    assert df_train.isnull().sum().sum() == 0, "Training set should not have any null values"
    assert df_test.isnull().sum().sum() == 0, "Test set should not have any null values"

    os.remove(f"{output_path }_train.csv")
    os.remove(f"{output_path }_test.csv")

