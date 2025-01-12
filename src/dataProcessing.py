import pickle
import pandas as pd

def load_data(file_path):
    """
    Loads hierarchical data from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: Loaded data in a hierarchical dictionary format.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def flatten_data(data):
    """
    Converts a hierarchical dictionary into a flat pandas DataFrame.

    Args:
        data (dict): Nested dictionary with the structure {syndrome_id -> subject_id -> image_id -> image_data}.

    Returns:
        pandas.DataFrame: A flat DataFrame with columns:
                         - syndrome_id
                         - subject_id
                         - image_id
                         - image_data
    """
    flattened_data = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, image_data in images.items():
                flattened_data.append({
                    "syndrome_id": syndrome_id,
                    "subject_id": subject_id,
                    "image_id": image_id,
                    "image_data": image_data
                })
    return pd.DataFrame(flattened_data)

def preprocess_data(df):
    """
    Cleans and preprocesses the flat DataFrame by converting key columns to numeric values.

    Args:
        df (pandas.DataFrame): Input DataFrame with hierarchical data in a flat structure.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame with numeric columns:
                         - syndrome_id
                         - subject_id
                         - image_id
    """
    df['syndrome_id'] = pd.to_numeric(df['syndrome_id'], errors='coerce')
    df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce')
    df['image_id'] = pd.to_numeric(df['image_id'], errors='coerce')
    return df