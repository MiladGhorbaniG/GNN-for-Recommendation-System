import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set the hyperparameters and train the model
embedding_dim = 16
hidden_dim = 32
dropout_p = 0.5

model = MovieLensNet(num_movies, num_users, num_genres, embedding_size=32, hidden_dim=64)
num_train_samples = len(train_loader.dataset)
print(f"Number of training samples: {num_train_samples}")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

number_epochs = 1  # Adjust the number of epochs as needed

for epoch in range(number_epochs):
    running_loss = 0.0
    for i, batch in tqdm(enumerate(train_loader)):
        movie_id = batch['movie_id']
        user_id = batch['user_id']
        genre_id = batch['genre_id']
        rating = batch['rating']

        output = model(movie_id, user_id, genre_id)
        loss = criterion(output, rating)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# Save the trained model if needed
# torch.save(model.state_dict(), 'model.pth')
