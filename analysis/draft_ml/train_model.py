from pathlib import Path
import time

import torch
import numpy as np

import lib.league_of_parquet as lop

# load dataset
DATASET_PATH = Path("../../data/dataset")
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


# define indices of the beginning and endo of the train and test sections
NUM_TOTAL_GAMES = DATASET.count_rows()
NUM_TEST_GAMES = NUM_TOTAL_GAMES // 10
RNG = np.random.default_rng(int(time.time()))

def take_random_games(num_games: int, train: bool):
    # generate a selection of random rows within the train/test sections
    row_indices = RNG.integers(low=NUM_TEST_GAMES if train else 0, high=NUM_TOTAL_GAMES if train else NUM_TEST_GAMES, size=num_games)

    # take rows from the dataset
    games = DATASET.take(row_indices).to_pandas()

    # load wins
    wins = games['win'].to_numpy().astype(dtype=np.float32)

    # load ranked scores
    ranked_scores = games['ranked_score'].to_numpy().astype(dtype=np.float32)

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
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    red_mask = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # select blue side targets
    blue_targets = picks[wins == 1]  # only select blue side wins
    blue_targets = blue_targets.reshape(-1, 5)[::2]  # only select the blue side picks
    blue_targets = blue_targets[np.arange(len(blue_targets)), np.tile(np.arange(5)[::-1], len(blue_targets) // 5)]  # select the picked champions in reverse order

    # select red side targets
    red_targets = picks[wins == 0]
    red_targets = red_targets.reshape(-1, 5)[1::2]
    red_targets = red_targets[np.arange(len(red_targets)), np.tile(np.arange(5)[::-1], len(red_targets) // 5)]

    # combine into one targets array
    all_targets = wins
    all_targets[all_targets == 1] = blue_targets
    all_targets[all_targets == 0] = red_targets

    # mask game picks for blue side drafts
    blue_side_picks = picks[wins == 1].ravel()
    mega_blue_mask = np.tile(blue_mask, (num_games, 1)).ravel()
    blue_side_picks[mega_blue_mask == 0] = 0

    # mask game picks for red side drafts
    red_side_picks = picks[wins == 0].ravel()
    mega_red_mask = np.tile(red_mask, (num_games, 1)).ravel()
    red_side_picks[mega_red_mask == 0] = 0

    # combine blue and red side drafts
    all_picks = picks
    all_picks[wins == 1] = blue_side_picks.reshape(-1, 10)
    all_picks[wins == 0] = red_side_picks.reshape(-1, 10)

    return np.concatenate((all_picks, bans, ranked_scores.reshape(-1, 1), wins.reshape(-1, 1)), axis=1), all_targets


print(take_random_games(10, False))
