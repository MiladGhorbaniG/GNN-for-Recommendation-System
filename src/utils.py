# utils.py
# Include any utility functions or classes that are used across your codebase here

import gdown
import zipfile
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import matplotlib.pyplot as plt

def download_and_extract_dataset(url, output_dir):
    gdown.download(url, output_dir, quiet=False)
    with zipfile.ZipFile(output_dir, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def load_movie_data(movie_data_path, rating_data_path):
    movies = pd.read_csv(movie_data_path)
    ratings = pd.read_csv(rating_data_path)

    # Clean the movies dataframe
    movies['year'] = movies['title'].str.extract('\((\d{4})\)', expand=False)
    movies['year'] = pd.to_datetime(movies['year'], format='%Y').dt.year
    movies['title'] = movies['title'].str.replace('(\(\d{4}\))', '').apply(lambda x: x.strip())

    # Filter the rating data to include only valid movie IDs
    valid_movie_ids = set(movies['movieId'])
    ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]

    # Fill missing values in ratings dataframe
