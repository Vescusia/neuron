import time
import random

from tqdm import tqdm
import torch
from sklearn.metrics import classification_report
import numpy as np

from model import NeuralNetwork
import lib.league_of_parquet as lop

DATASET_PATH = "../../data/dataset"

DATASET = lop.open_dataset(DATASET_PATH)

TOTAL_ROWS = DATASET.count_rows()
TEST_ROWS = TOTAL_ROWS // 10


def take_random_games(num_rows: int, train: bool):
    rows = []
    # generate random row indexes within the train/test ranges
    for _ in range(num_rows):
        rows.append(random.randint(TEST_ROWS if train else 0, TOTAL_ROWS - 1 if train else TEST_ROWS - 1))

    # take the rows from the dataset and covert them to python objects
    batch = DATASET.take(rows).to_pydict()
    # initialize a list for each game
    games = [[] for _ in range(num_rows)]

    # iterate over the columns and append the partial data to each game
    for i, column in enumerate(batch.keys()):
        # skip the patch column (wanna be patch-agnostic)
        if column == "patch":
            continue
        # skip the win column (we're predicting it)
        elif column == "win":
            wins = [1 if win else 0 for win in batch[column]]
            continue
        # append ranked score, picks and bans to the game
        for j, entry in enumerate(batch[column]):
            games[j].append(entry)

    return games, np.array(wins, dtype=np.float32)


def train_model(batch_size=10_000, evaluate_every=25):
    # train loop
    model = NeuralNetwork(172)  # HAS TO BEEEEE ONE LARGER BECAUSE OF NO CHAMP IN BANS
    model.fit_scaler(take_random_games(10000, True)[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    start_time = time.time()
    for epoch in range(10):
        total_loss = 0.0

        # train model for this epoch
        batch_range = tqdm(range((TOTAL_ROWS - TEST_ROWS) // batch_size))
        for i in batch_range:
            # get a batch of random games
            games, wins = take_random_games(batch_size, True)
            games = model.embed_games(games)
            games = torch.from_numpy(games)
            wins = torch.from_numpy(wins)

            # predict win/lose
            predicted_wins = torch.flatten(model(games))
            # calculate loss
            loss = loss_fn(predicted_wins, wins)
            total_loss += loss.item()
            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % evaluate_every == 0 and i != 0:
                # evaluate model
                model.eval()
                with torch.no_grad():
                    # get a batch of random games
                    games, wins = take_random_games(batch_size, False)
                    games = model.embed_games(games)
                    games = torch.from_numpy(games)

                    # predict win/lose
                    predicted_wins = torch.flatten(model(games))
                    predicted_wins = predicted_wins.numpy()
                    predicted_wins = np.round(predicted_wins)

                    print("\n" + classification_report(wins, predicted_wins, zero_division=np.nan)
                          + f"total loss: {total_loss:.4f} in {(time.time() - start_time) / 60:.2f} m (EPOCH {epoch + 1})"
                          )

                model.train()


if __name__ == "__main__":
    train_model()
