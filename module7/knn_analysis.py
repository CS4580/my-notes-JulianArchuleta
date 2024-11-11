"""KNN Analysis of Movies
"""
import pandas as pd
import numpy as np
import get_data as gt  # your package
import Levenshtein  # Levenshtein distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
K = 10  # number of closest matches
BASE_CASE_ID = 88763  # IMDB_id for 'Back to the Future'
SECOND_CASE_ID = 89530  # IMDB_id for 'Mad Max Beyond Thunderdome'
BASE_YEAR = 1985  # year for 'Back to the Future'

METRIC1_WT = 0.2  # weight for cosine similarity
METRIC2_WT = 0.8  # weight for weighted Jaccard similarity


def metric_stub(base_case_value, comparator_value):
    """
    Dummy metric function
    base_case_value (any type): The base value to compare against.
    comparator_value (any type): The value to compare with the base case.
    Returns:
    int: Always returns 0 as this is a stub function.
    """

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


def _get_weighted_jaccard_similarity_dict(df):
    # put our selections of 'Back to the Future'(88763) and 'Mad Max Beyond Thunderome'(89530) into a list:
    selections_df = [df.loc[BASE_CASE_ID], df.loc[SECOND_CASE_ID]]

    # genres_weighted_dictionary is needed for the weighted Jaccard similarity index:
    genres_weighted_dictionary = {"total": 0}
    for movie in selections_df:
        # the genres are separated by a semicolon
        for genre in movie["genres"].split(";"):
            if genre in genres_weighted_dictionary:
                genres_weighted_dictionary[genre] += 1
            else:
                genres_weighted_dictionary[genre] = 1
            genres_weighted_dictionary["total"] += 1

    # print(f"\t**genres_weighted_dictionary = {genres_weighted_dictionary}")
    return genres_weighted_dictionary


def weighted_jaccard_similarity(df: pd.DataFrame, comparator_genres: str):
    # weighted_dictionary is based on all the selections that the user has made so far
    # comparator_genres is another movie's genres that is being compared
    weighted_dictionary = _get_weighted_jaccard_similarity_dict(df)
    numerator = 0
    denominator = weighted_dictionary["total"]
    for genre in comparator_genres.split(";"):
        if genre in weighted_dictionary:
            numerator += weighted_dictionary[genre]

    return numerator / denominator


def cosine_similarity_function(base_case_plot, comparator_plot):
    # this line will convert the plots from strings to vectors in a single matrix:
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        (base_case_plot, comparator_plot))
    results = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return results[0][0]


def cosine_and_weighted_jaccard(df: pd.DataFrame, plots: str, comparator_movie: pd.core.series.Series,):
    # Perform the cosine similiarty and weighted Jaccard metrics:
    cs_result = cosine_similarity_function(plots, comparator_movie["plot"])
    weighted_dictionary = _get_weighted_jaccard_similarity_dict(df)
    wjs_result = weighted_jaccard_similarity(
        df, comparator_movie["genres"]
    )

    # Normalization:
    # The weighted Jaccard similarity result has a range from 0.0 to 1.0.
    # The cosine similarity result has a range from -1.0 to 1.0. We need to change the range for the cosine similarity result.
    # First, add 1 to the cosine similarity result so that it has a range from 0.0 to 2.0
    # Second, divide the result by 2.0 so that it has a range from 0.0 to 1.0:
    cs_result = (cs_result + 1) / 2.0

    # Weights:
    # Use a weight of 0.2 (20%) for the cosine similarity result:
    cs_result *= METRIC1_WT
    # Use a weight of 0.8 (80%) for the weighted Jaccard similarity result:
    wjs_result *= METRIC2_WT
    return wjs_result + cs_result


def knn_analysis_driver(data_df, base_case, comparison_type, metric_func, sorted_value='metric'):
    df = data_df.copy()
    # WIP: Create df of filter data
    df[sorted_value] = df[comparison_type].map(
        lambda x: metric_func(base_case[comparison_type], x))
    # Sort return values from functio, stub
    if 'jaccard' in metric_func.__name__:
        sorted_df = df.sort_values(by=sorted_value, ascending=False)
    else:
        sorted_df = df.sort_values(by=sorted_value)  # default is ascending
    sorted_df = sorted_df.drop(BASE_CASE_ID)  # drop base case
    # Print top 10 values
    print_top_k(sorted_df, sorted_value, comparison_type)


def main():
    # TASK 1: Get dataset from server
    print(f'Task 1: Download dataset from server')
    dataset_file = 'movies.csv'
    gt.download_dataset(gt.ICARUS_CS4580_DATASET_URL, dataset_file)

    # TASK 2: Load  data_file into a DataFrame
    print(f'\nTask 2: Load movie data into a DataFrame')
    data_file = f'{gt.DATA_FOLDER}/{dataset_file}'
    data = gt.load_data(data_file, index_col='IMDB_id')
    print(f'Loaded {len(data)} records')
    print(f'Data set Columns {data.columns}')
    print(f'Data set description {data.describe()}')

    # TASK 3: KNN Analysis
    print(f'\nTask 3: KNN Analysis')
    base_case = data.loc[BASE_CASE_ID]
    print(f'Comparing all movies to our base case: {base_case["title"]}')
    knn_analysis_driver(data_df=data, base_case=base_case, comparison_type='genres',
                        metric_func=metric_stub, sorted_value='metric')

    # TASK 4: KNN Analysis with Euclidean Distance
    print(f'\nTask 4: KNN Analysis with Euclidean Distance')
    knn_analysis_driver(data_df=data, base_case=base_case, comparison_type='year',
                        metric_func=euclidean_distance, sorted_value='euclidean_distance')

    # TASK 5: Jaccard Similarity
    print(f'\nTask 5: KNN Analysis with Jaccard Similarity')
    data = data[data['year'] >= BASE_YEAR]  # added filter
    knn_analysis_driver(data_df=data, base_case=base_case, comparison_type='genres',
                        metric_func=jaccard_similarity_normal, sorted_value='jaccard_similarity')

    # Task 6: KNN Analysis with Weighted Jaccard Similarity
    print(f"\nTask 6: KNN Analysis with Weighted Jaccard Similarity")
    base_case = data.loc[BASE_CASE_ID]  # base case
    second_case = data.loc[SECOND_CASE_ID]  # second case
    print(
        f"Comparing all movies to our base case: {
            base_case['title']} and {second_case['title']}."
    )
    # Add two additional filters: stars >= 5 and rating = ['G', 'PG', 'PG-13']
    data = data[(data["stars"] >= 5) & (
        data["rating"].isin(["G", "PG", "PG-13"]))]
    knn_analysis_driver(
        data,
        base_case,
        comparison_type="genres",
        metric_func=weighted_jaccard_similarity,
        sorted_value="jaccard_similarity",
    )

    # Task 7: KNN Analysis with Levenshtein Distance
    print(f"\nTask 7: KNN Analysis with Levenshtein Distance")
    # reload data to remove filters
    data = gt.load_data(data_file, index_col="IMDB_id")
    base_case = data.loc[BASE_CASE_ID]  # base case
    print(f"Comparing all movies to our base case: {base_case['title']}.")
    knn_analysis_driver(
        data,
        base_case,
        comparison_type="title",
        metric_func=Levenshtein.distance,
        sorted_value="levenshtein_distance",
    )

    # Task 8: KNN Analysis with Cosine Similarity
    print(f'\nTask 8: KNN Analysis with Cosine Similarity')
    knn_analysis_driver(data, base_case, comparison_type='plot',
                        metric_func=cosine_similarity_function, sorted_value='cosine_similarity')
    # Task 9: KNN Analysis with Cosine and Weighted Jaccard
    print(f'\nTask 9: KNN Analysis with Cosine and Weighted Jaccard')
    # Add filters
    data = data[data['year'] >= BASE_YEAR]  # filter by year 1985 and above
    data = data[(data['stars'] >= 5) & (
        data['rating'].isin(['G', 'PG', 'PG-13']))]
    knn_analysis_driver(data, base_case, comparison_type='genres',
                        metric_func=cosine_and_weighted_jaccard, sorted_value='cosine_and_weighted_jaccard')


if __name__ == '__main__':
    main()
