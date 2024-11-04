"""KNN Analysis of Movies
"""
import pandas as pd
import numpy as np
import get_data as gt  # your package

# Constants
K = 10  # number of closest matches
BASE_CASE_ID = 88763  # IMDB_id for 'Back to the Future'
BASE_YEAR = 1985  # year for 'Back to the Future'


def metric_stub(base_case_value, comparator_value):
    return 0


def print_top_k(df, sorted_value, comparison_type):
    print(f'Top {K} closest matches to {comparison_type}')
    counter = 1
    for idx, row in df.head(K).iterrows():
        print(f'Top {counter} match:[{idx}]:{row["year"]}, {
              row["title"]}, {row["genres"]}, [{row[sorted_value]}]')
        counter += 1


def euclidean_distance(base_case_year: int, comparator_year: int):
    return abs(base_case_year - comparator_year)


def jaccard_similarity_normal(base_case_genres: str, comparator_genres: str):
    base_case_set = set(base_case_genres.split(';'))
    comparator_set = set(comparator_genres.split(';'))
    intersection = len(base_case_set.intersection(comparator_set))
    union = len(base_case_set.union(comparator_set))
    return float(intersection / union)


def knn_analysis_driver(data_df, base_case, comparison_type, metric_func, sorted_value='metric'):
    df = data_df.copy()
    # WIP: Create df of filter data
    df[sorted_value] = df[comparison_type].map(
        lambda x: metric_func(base_case[comparison_type], x))
    # Sort return values from functio, stub
    if 'jaccard' in metric_func.__name__:
        sorted_df = df.sort_values(by=sorted_value, ascending=False)
    else:
        sorted_df = df.sort_values(by=sorted_value) # default is ascending
    sorted_df = sorted_df.drop(BASE_CASE_ID) # drop base case
    # Print top 10 values
    print_top_k(sorted_df, sorted_value, comparison_type)


def main():
    # TASK 1: Get dataset from server
    print(f'Task 1: Download dataset from server')
    dataset_file = 'movies.csv'
    gt.download_dataset(gt.ICARUS_CS4580_DATASET_URL, dataset_file)
    # TASK 2: Load  data_file into a DataFrame
    print(f'Task 2: Load movie data into a DataFrame')
    data_file = f'{gt.DATA_FOLDER}/{dataset_file}'
    data = gt.load_data(data_file, index_col='IMDB_id')
    print(f'Loaded {len(data)} records')
    print(f'Data set Columns {data.columns}')
    print(f'Data set description {data.describe()}')
    # TASK 3: KNN Analysis
    print(f'Task 3: KNN Analysis')
    base_case = data.loc[BASE_CASE_ID]
    print(f'Comparing all movies to our base case: {base_case["title"]}')
    knn_analysis_driver(data_df=data, base_case=base_case, comparison_type='genres',
                        metric_func=metric_stub, sorted_value='metric')
    # TASK 4: KNN Analysis with Euclidean Distance
    print(f'Task 4: KNN Analysis with Euclidean Distance')
    knn_analysis_driver(data_df=data, base_case=base_case, comparison_type='year',
                        metric_func=euclidean_distance, sorted_value='euclidean_distance')
    # TASK 5: Jaccard Similarity
    print(f'Task 5: KNN Analysis with Jaccard Similarity')
    data = data[data['year'] >= BASE_YEAR]  # added filter
    knn_analysis_driver(data_df=data, base_case=base_case, comparison_type='genres',
                        metric_func=jaccard_similarity_normal, sorted_value='jaccard_similarity')


if __name__ == '__main__':
    main()
