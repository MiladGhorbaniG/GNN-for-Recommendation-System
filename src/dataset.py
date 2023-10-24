import torch
from torch.utils.data import Dataset

class MovieLensDataset(torch.utils.data.Dataset):
    def __init__(self, data, movies, genres_encoded, mlb, max_genre_count, num_users):
        self.data = data
        self.movies = movies
        self.genres_encoded = genres_encoded
        self.mlb = mlb
        self.max_genre_count = max_genre_count
        self.num_users = num_users

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        movie_id = torch.tensor(row["movieId"], dtype=torch.long)
        user_id = torch.tensor(row["userId"], dtype=torch.long)
        if user_id.min() < 0 or user_id.max() >= self.num_users:
            print('self.num_users = ' , self.num_users)
            raise ValueError(f"Invalid user ID: {user_id}")
        movie_genres = self.movies.loc[self.movies['movieId'] == row['movieId'], 'genres'].iloc[0]
        genre_indices = []
        for genre in movie_genres.split('|'):
            if genre in self.mlb.classes_:
                genre_indices.append(np.where(self.mlb.classes_ == genre)[0][0])
        if len(genre_indices) == 0:  # Add a default value for an empty genre tensor
            genre_indices.append(0)
        genre_id = torch.tensor(genre_indices, dtype=torch.long)
        genre_id = torch.flatten(genre_id)[:self.max_genre_count]
        genre_pad = torch.zeros(self.max_genre_count - genre_id.shape[0], dtype=torch.long)
        genre_id = torch.cat([genre_id, genre_pad])
        rating = torch.tensor(row["rating"], dtype=torch.float)
        return {"movie_id": movie_id, "user_id": user_id, "genre_id": genre_id, "rating": rating}
