import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Evaluate the model on the test set
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model.eval()
with torch.no_grad():
    total_loss = 0.0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Collect the predicted ratings and true ratings
    all_predicted_ratings = []
    all_true_ratings = []

    for i, batch in enumerate(test_loader):
        if i + 1 == stop_i:
            break
        movie_id = batch['movie_id']
        user_id = batch['user_id']
        genre_id = batch['genre_id']
        rating = batch['rating']
        output = model(movie_id, user_id, genre_id)
        loss = criterion(output, rating)
        total_loss += loss.item()
        predicted_labels = (output >= 2).float()  # Threshold at 2 or higher
        tp += ((predicted_labels == 1) & (rating >= 3.5)).sum().item()
        fp += ((predicted_labels == 1) & (rating < 3.5)).sum().item()
        tn += ((predicted_labels == 0) & (rating < 3.5)).sum().item()
        fn += ((predicted_labels == 0) & (rating >= 3.5)).sum().item()

        all_predicted_ratings.extend(output.squeeze().tolist())
        all_true_ratings.extend(rating.squeeze().tolist())

    # Plot distribution of true and predicted ratings
    plt.hist(all_true_ratings, bins=np.arange(0, 6, 0.5), alpha=0.5, label='True ratings')
    plt.hist(all_predicted_ratings, bins=np.arange(0, 6, 0.5), alpha=0.5, label='Predicted ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    # Plot distribution of prediction errors
    errors = np.array(all_true_ratings) - np.array(all_predicted_ratings)
    plt.hist(errors, bins=np.arange(-3, 4, 0.5))
    plt.xlabel 'Prediction error')
    plt.ylabel('Count')
    plt.show()

    avg_loss = total_loss / len(test_loader)
    print('Test RMSE: %.3f' % np.sqrt(avg_loss))
    precision = tp / (tp + fp + 0.000001) * 100
    recall = tp / (tp + fn + 0.000001) * 100
    f1 = 2 * precision * recall / (precision + recall + 0.000001)
    print('Test precision: %.3f' % precision + ' %')
    print('Test recall: %.3f' % recall + ' %')
    print('Test F1 score: %.3f' % f1 + ' %')

    # Compute the optimal threshold
    thresholds = np.arange(2, 5.0, 0.1)
    f1_scores = []
    for threshold in thresholds:
        predicted_labels = (torch.Tensor(all_predicted_ratings) >= threshold).float()
        tp = ((predicted_labels == 1) & (torch.Tensor(all_true_ratings) >= 3.5)).sum().item()
        fp = ((predicted_labels == 1) & (torch.Tensor(all_true_ratings) < 3.5)).sum().item()
        tn = ((predicted_labels == 0) & (torch.Tensor(all_true_ratings) < 3.5)).sum().item()
        fn = ((predicted_labels == 0) & (torch.Tensor(all_true_ratings) >= 3.5)).sum().item()
        precision = tp / (tp + fp + 0.000001)
        recall = tp / (tp + fn + 0.000001)
        f1 = 2 * precision * recall / (precision + recall + 0.000001)
        f1_scores.append(f1)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    # print('Optimal threshold:', optimal_threshold)
