import torch
import torch.nn as nn

class MovieLensNet(nn.Module):
    def __init__(self, num_movies, num_users, num_genres_encoded, embedding_size, hidden_dim):
        super(MovieLensNet, self).__init__()
        self.num_movies = num_movies
        self.num_users = num_users
        self.num_genres_encoded = num_genres_encoded
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2 + num_genres_encoded, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, movie_id, user_id, genre_id):
        genre_id = torch.unsqueeze(genre_id, dim=2)
        if genre_id.size() != (movie_id.size(0), self.num_genres_encoded, 1):
            raise ValueError(f"Expected genre_id to have size ({movie_id.size(0)}, {self.num_genres_encoded}, 1), but got size {genre_id.size()}")
        movie_emb = self.movie_embedding(movie_id)
        user_emb = self.user_embedding(user_id)
        movie_emb = torch.unsqueeze(movie_emb, dim=2)
        user_emb = torch.unsqueeze(user_emb, dim=2)
        x = torch.cat([movie_emb, user_emb, genre_id.float()], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
