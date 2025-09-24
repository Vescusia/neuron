from pathlib import Path
import time

import sklearn
import torch
import numpy as np

import lib.league_of_parquet as lop
from .model import ResNet20,Embedder


# load dataset
DATASET_PATH = Path("./data/dataset")
print(f"Opening Dataset from {DATASET_PATH}")
DATASET = lop.open_dataset(DATASET_PATH)

# define save directory for the model
MODEL_SAVE_DIR = Path("./analysis/draft_ml/models")

# check for CUDA availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    DEVICE = torch.device("cuda")
else:
    print("No GPU available.")
    DEVICE = torch.device("cpu")


# define indices of the beginning and end of the train and test sections
NUM_TOTAL_GAMES = DATASET.count_rows()
NUM_TEST_GAMES = NUM_TOTAL_GAMES // 10
RNG = np.random.default_rng(int(time.time()))


def take_random_games(num_games: int, train: bool):
    # generate a selection of random rows within the train/test sections
    row_indices = RNG.integers(low=NUM_TEST_GAMES if train else 0, high=NUM_TOTAL_GAMES if train else NUM_TEST_GAMES, size=num_games)

    # take rows from the dataset
    games = DATASET.take(row_indices)

    # load wins
    wins = games['win'].to_numpy().astype(dtype=np.uint16)

    # load ranked scores
    ranked_scores = games['ranked_score'].to_numpy().astype(dtype=np.uint16)

    # load picks
    picks = np.array(games['picks'].to_numpy().tolist(), dtype=np.uint16)

    # load bans
    bans = np.array(games['bans'].to_numpy().tolist(), dtype=np.uint16)

    # repeat games by 5, as there are 5 picks per game
    wins = np.repeat(wins, 5, axis=0)
    ranked_scores = np.repeat(ranked_scores, 5, axis=0)
    picks = np.repeat(picks, 5, axis=0)
    bans = np.repeat(bans, 5, axis=0)

    # create draft masks
    blue_mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    ])
    red_mask = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ])

    # select blue side targets
    blue_targets = picks[wins == 1]  # only select blue side wins
    blue_targets = blue_targets.reshape(-1, 5)[::2]  # only select the blue side picks
    blue_targets = blue_targets[np.arange(len(blue_targets)), np.tile(np.arange(5), len(blue_targets) // 5)]  # select the picked champions in reverse order

    # select red side targets
    red_targets = picks[wins == 0]
    red_targets = red_targets.reshape(-1, 5)[1::2]
    red_targets = red_targets[np.arange(len(red_targets)), np.tile(np.arange(5), len(red_targets) // 5)]

    # combine into one targets array
    all_targets = np.empty(num_games * 5, dtype=np.uint16)
    all_targets[wins == 1] = blue_targets
    all_targets[wins == 0] = red_targets

    # mask game picks for blue side drafts
    blue_side_picks = picks[wins == 1].ravel()
    mega_blue_mask = np.tile(blue_mask, (len(blue_side_picks) // 50, 1)).ravel()
    blue_side_picks[mega_blue_mask == 0] = 0

    # mask game picks for red side drafts
    red_side_picks = picks[wins == 0].ravel()
    mega_red_mask = np.tile(red_mask, (len(red_side_picks) // 50, 1)).ravel()
    red_side_picks[mega_red_mask == 0] = 0

    # combine blue and red side drafts
    all_picks = picks
    all_picks[wins == 1] = blue_side_picks.reshape(-1, 10)
    all_picks[wins == 0] = red_side_picks.reshape(-1, 10)

    # concatenate into one games array
    games = np.concatenate((all_picks, bans, ranked_scores.reshape(-1, 1), wins.reshape(-1, 1)), axis=1)

    return games, all_targets


def train_model(batch_size: int = 10_000, evaluate_every: int = 10_000_000):
    # create model
    params = {"num_champions": 171, "width": 128, "bottleneck": 5, "dropout": 0.1, "blocks_pre_win": 10, "blocks_pre_rank": 10, "blocks_post_rank": 20}
    model = ResNet20(**params).to(DEVICE)
    # create embedder
    embedder = Embedder(params["num_champions"])
    embedder.fit(take_random_games(batch_size, True)[0])

    # store models and reports over time
    models = []
    reports = []

    # create optimizer and loss
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.NLLLoss()

    # train model
    try:
        games_trained = 0
        last_report = 0
        total_loss = 0

        while True:
            # get a new batch
            games, picks = take_random_games(batch_size, True)
            games_trained += len(games)

            # embed the batch
            games = embedder(games)
            games = torch.from_numpy(games).to(DEVICE)

            # predict
            pred_picks = model(games)

            # calculate loss
            picks = torch.from_numpy(picks).to(DEVICE)
            print(picks)
            loss = loss_fn(pred_picks, picks)
            total_loss = loss.item()

            # backpropagate
            loss.backward()
            optim.zero_grad()
            optim.step()

            # evaluate every once in a while
            if games_trained - last_report >= evaluate_every:
                model.eval()
                with torch.no_grad():
                    # get a new batch
                    games, picks = take_random_games(batch_size, False)
                    games_trained += len(games)

                    # embed the batch
                    games = embedder(games)
                    games = torch.from_numpy(games).to(DEVICE)

                    # predict
                    pred_picks = model(games)
                    # take maximum argument as prediction
                    pred_picks = torch.argmax(pred_picks, dim=1).cpu().numpy()

                    # report
                    print(sklearn.metrics.classification_report(picks, pred_picks))
                    print(f"total loss since last report: {total_loss}")

                model.train()
                total_loss = 0

    finally:
        pass
