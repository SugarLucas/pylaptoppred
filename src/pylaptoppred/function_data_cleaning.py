import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_and_save_data_f(input_file, output_file):

    """
    Reads data from a specified input file, performs data cleaning by removing outliers using the IQR method on numeric columns,
    drops duplicate rows, handles missing values by dropping rows with missing data, and saves the cleaned data to a specified output file.

    Parameters:
    -----------
    input_path: str
        Path/filename of the data to be read in.

    output_path: str
        Path/filename where the cleaned/processed data will be saved.

    Returns:
    --------
    tuple of DataFrame
        A tuple containing one training dataframe and one testing dataframe.
        
    Examples:
    ---------
    >>> train, test = clean_and_save_data_f('inputFilePath.csv', 'newFilePath') # replace ('inputFilePath', 'newFilePath') with the actual filepath of data and filepath to be written to
    >>> print(train.head())
    >>> print(test.head())

    Notes:
    ------
    This function requires pandas library

    """
    # Read the data
    df = pd.read_csv(input_file)

    # Basic data cleaning
    # Drop rows where any data is missing
    df_cleaned = df.dropna()

    # Drop duplicate rows, keep the first occurrence
    df_cleaned = df_cleaned.drop_duplicates()

    # Removing outliers using the IQR method, but only for numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    Q1 = df_cleaned[numeric_cols].quantile(0.25)
    Q3 = df_cleaned[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Define a mask to filter out outliers only for numeric columns
    filter = ~((df_cleaned[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_cleaned[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_final = df_cleaned[filter]

    # Save the cleaned data
    df_train, df_test = train_test_split(df_final, test_size=0.3, random_state=42)

    # Save the training and test sets to separate files
    train_output_file = f"{output_file}_train.csv"
    test_output_file = f"{output_file}_test.csv"
    df_train.to_csv(train_output_file, index=False)
    df_test.to_csv(test_output_file, index=False)

    print(f"Data cleaned and saved to {output_file}_train.csv and {output_file}_test.csv")

    # Return the training and test dataframes
    return df_train, df_test
