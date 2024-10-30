"""KNN Analysis of Movies
"""
import pandas as pd
import numpy as np
import get_data as gt # your package

# Constants
k = 10
BASE_CASE_ID = 88763 # IMDB_id for 'Back to the Future'

def knn_analysis_driver(df,base_case, compairson_type, metric_stub, sorted_values='metric'):
    # WIP: Create df of filter data
    df[sorted_values] = df[compairson_type].map(lambda x: metric_stub(base_case[compairson_type], x))

def main():
    # Task 1: Get dataset from server
    print("Task 1: Download dataset from server")
    dataset_file = 'movies.csv'
    gt.download_data(gt.URL, dataset_file)
    # Task 2: Load data_file into dataframe
    print("Task 2: Load data_file into dataframe")
    data_file = f'{gt.DATA_FOLDER}/{dataset_file}'
    data = gt.load_data(data_file, index_col='IMDB_ID')
    print(f'Loaded {len(data)} records')
    print(f'Data set Columns: {data.columns}')
    print(f'Data set description: {data.describe()}')
    # TODO: The rest of your code goes here
    
if __name__ == '__main__':
    main()