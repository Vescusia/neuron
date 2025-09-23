import time

from tqdm import tqdm
import torch
import sklearn
import numpy as np

from .model import Embedder, LinearWide54, ResNet60
import lib.league_of_parquet as lop

DATASET_PATH = "./data/dataset"
print(f"Opening Dataset from {DATASET_PATH}")
DATASET = lop.open_dataset(DATASET_PATH)


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    DEVICE = torch.device("cuda")
else:
    print("No GPU available.")
    DEVICE = torch.device("cpu")


def load_dataset():
    pd_dataset = DATASET.to_table().to_pandas()

    wins = pd_dataset["win"].to_numpy().astype(np.float32)
    games = pd_dataset[["ranked_score", "picks", "bans"]].to_numpy()

    return games, wins


def train_model(batch_size=10_000, evaluate_every=500_000):
    # load dataset
    start = time.time()
    train_games, train_wins = load_dataset()
    train_games, test_games, train_wins, test_wins = sklearn.model_selection.train_test_split(train_games, train_wins, test_size=0.10)
    print(f"Loaded Dataset in {(time.time() - start):.0f} s")

    # initialize model and optimizer
    model = ResNet60(171, 0.1, 2, 128).to(DEVICE)
    print(f"Model has {sum(p.numel() for p in model.parameters()):_} parameters")
    embedder = Embedder(171)
    embedder.fit(test_games[0:batch_size])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    # train loop
    start_time = time.time()
    total_loss = 0.0
    for epoch in range(10):
        # train model for this epoch
        batch_range = tqdm(range(0, len(train_games), batch_size))
        batch_range.set_description(f"EPOCH {epoch + 1}")
        for i in batch_range:
            # get a batch of random games
            games, wins = train_games[i:i + batch_size], train_wins[i:i + batch_size]
            games = embedder(games)
            games = torch.from_numpy(games).to(DEVICE)
            wins = torch.from_numpy(wins).to(DEVICE)

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
                    games, wins = test_games[0:batch_size], test_wins[0:batch_size]
                    games = embedder(games)
                    games = torch.from_numpy(games).to(DEVICE)

                    # predict win/lose
                    predicted_wins = torch.flatten(model(games))
                    predicted_wins = predicted_wins.cpu().numpy()

                    print(f"\nAlpha of block 1: {model.get_parameter('res_block_stack.2.alpha')}")
                    print(sklearn.metrics.classification_report(wins, np.round(predicted_wins), zero_division=np.nan)
                          + f"total loss: {total_loss:.4f} in {(time.time() - start_time) / 60:.2f} m (EPOCH {epoch + 1})"
                          )

                    rand_array = np.random.rand(len(predicted_wins))
                    for j in range(len(predicted_wins)):
                        if rand_array[j] < predicted_wins[j]:
                            predicted_wins[j] = 0
                        else:
                            predicted_wins[j] = 1

                    print("\n" + sklearn.metrics.classification_report(wins, predicted_wins, zero_division=np.nan))

                total_loss = 0.0
                model.train()

        # reshuffle train and test data (separately!) every epoch
        train_games, train_wins = sklearn.utils.shuffle(train_games, train_wins)
        test_games, test_wins = sklearn.utils.shuffle(test_games, test_wins)


if __name__ == "__main__":
    train_model()
