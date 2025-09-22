import lib.league_of_parquet as lop
import torch
import random
from model import NeuralNetwork
from sklearn.metrics import classification_report
import time
import numpy as np


path = "../../data/dataset"

dataset = lop.open_dataset(path)

total_rows = dataset.count_rows()
test_rows = total_rows // 10

def take_random_games(num_rows: int, train: bool):
    rows = []
    for _ in range(num_rows):
        rows.append(random.randint(0 if not train else test_rows, total_rows - 1 if train else test_rows - 1))

    batch = dataset.take(rows).to_pydict()
    games = [[] for _ in range(num_rows)]

    for i, column in enumerate(batch.keys()):
        if column == "patch":
            continue
        if column == "win":
            wins = [[[1,0]] if win else [[0,1]] for win in batch[column]]
            continue
        for j, entry in enumerate(batch[column]):
            games[j].append(entry)

    return games, wins


# train loop
model = NeuralNetwork(172) # HAS TO BEEEEE ONE LARGER BECAUSE OF NO CHAMP IN BANS
model.fit_scaler(take_random_games(10000, True)[0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()

start_time = time.time()
for epoch in range(10):
    total_loss = 0.0
    print(f"Epoch {epoch}")
    epoch_start_time = time.time()

    for _ in range(total_rows // 10000):
        games, targets = take_random_games(10000, True)
        games = model.embed_games(games)
        games = np.array(games, dtype=np.float32)
        games = torch.from_numpy(games)
        targets = torch.tensor(targets, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(games)
        print(outputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            games, targets = take_random_games(10000, True)
            games = model.embed_games(games)
            games = np.array(games, dtype=np.float32)
            games = torch.from_numpy(games)

            outputs = model(games)

            print(classification_report(np.array(targets).argmax(), np.array(outputs), zero_division=np.nan), end="")

            total_elapsed = time.time() - start_time
            epoch_elapsed = time.time() - epoch_start_time
            epoch_start_time = time.time()
            print(
                f"total_loss: {total_loss:.2f} in {total_elapsed / 60:.2f} m ({epoch_elapsed:.2f} s for last epoch) | {epoch + 1}/10\n")

        model.train()
