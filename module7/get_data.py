"""
CS 4580 - Assignment 4
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.stats import pearsonr, spearmanr
DATA_FOLDER = 'data'
URL = 'icarus.cs.weber.edu:~hvalle/cs4580/data'


def download_data(url, data_path, data_folder=DATA_FOLDER):
    import urllib.request

    # Ensure the data folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Construct the full path to save the file
    full_url = URL + '/' + data_path
    full_path = data_folder + '/' + data_path

    # Download the file from the URL
    try:
        urllib.request.urlretrieve(full_url, full_path)
        print(f'Data downloaded successfully and saved to {full_path}')
    except Exception as e:
        print(f'Failed to download data: {e}')
        raise
    

def load_data(file_path, index_col=None):
    """
    Loads data from a CSV file

    Args:
        file_path (str): The path to the CSV file containing the census data.'
        index_col (str): The column to use as the index for the DataFrame

    Returns:
        pandas.DataFrame: A DataFrame containing the weather data
    """
    # Check if file is in csv format
    if not file_path.endswith('.csv'):
        print('File must be in CSV format')
        raise ValueError('File must be in CSV format')
    # Check if data is a valid file path or raise an error
    if not os.path.exists(file_path):
        print('File not found')
        raise FileNotFoundError('File not found')
    
    # Load the data
    if index_col:
        df = pd.read_csv(file_path, index_col=index_col)
    else:
        df = pd.read_csv(file_path)
    
    return df


def main():
    # TASK 1: Load the data
    print(f'Task 1: Download dataset from server')
    dataset_file = 'movies.csv'
    download_data(URL, dataset_file)
    # TASK 2: Load data_file into dataframe
    print(f'Task 2: Load data from {dataset_file}')
    data_file = f'{DATA_FOLDER}/{dataset_file}'
    data = load_data(data_file, index_col='IMDB_ID')
    print(f'Loaded {len(data)} records')

if __name__ == '__main__':
    main() 