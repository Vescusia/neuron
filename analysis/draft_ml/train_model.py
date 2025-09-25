from pathlib import Path
import time
from json import dumps

import sklearn
import torch
import numpy as np
from tqdm import tqdm
from pyarrow import RecordBatch

import lib.league_of_parquet as lop
from .model import ResNet20,Embedder


# define dataset path
DATASET_PATH = Path("./data/dataset")

# define save directory for the model
MODEL_SAVE_DIR = Path("./analysis/draft_ml/models")

# check for CUDA availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    DEVICE = torch.device("cuda")
else:
    print("No GPU available.")
    DEVICE = torch.device("cpu")

# create RNG instance
RNG = np.random.default_rng(int(time.time()))


def parse_batch(batch: RecordBatch):
    batch = batch.to_pandas()

    # load wins
    wins = batch['win'].to_numpy().astype(dtype=np.uint16)

    # load ranked scores
    ranked_scores = batch['ranked_score'].to_numpy().astype(dtype=np.uint16)

    # load picks
    picks = np.array(batch['picks'].to_numpy().tolist(), dtype=np.uint16)

    # load bans
    bans = np.array(batch['bans'].to_numpy().tolist(), dtype=np.uint16)

    # shuffle them together
    shuffle_indices = np.arange(len(batch))
    RNG.shuffle(shuffle_indices)
    wins = wins[shuffle_indices]
    ranked_scores = ranked_scores[shuffle_indices]
    picks = picks[shuffle_indices]
    bans = bans[shuffle_indices]

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
    all_targets = np.empty(len(batch) * 5, dtype=np.int64)
    all_targets[wins == 1] = blue_targets
    all_targets[wins == 0] = red_targets
    all_targets -= 1  # only no pick/ban is 0

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


def train_model(batch_size: int = 24_000, evaluate_every: int = 10_000_000, test_split: float = 0.1):
    # load dataset
    dataset = lop.open_dataset(DATASET_PATH)
    print(f"Opened dataset from {DATASET_PATH}")

    # determine batches that belong to training
    num_test_games = int(dataset.count_rows() * test_split)
    num_test_batches = 0
    for i, batch in enumerate(dataset.to_batches()):
        num_test_batches += len(batch)  # just use the test batches variable as temporary storage

        if num_test_batches >= num_test_games:
            num_test_batches = i
            break

    # create model
    params = {"num_champions": 171, "width": 256, "bottleneck": 16, "dropout": 0.5, "blocks_pre_win": 4, "blocks_pre_rank": 4, "blocks_post_rank": 8}
    model = ResNet20(**params).to(DEVICE)
    print(f"Model has {sum(p.numel() for p in model.parameters()):_} parameters")

    # create embedder
    embedder = Embedder(params["num_champions"])
    embedder.fit(parse_batch(dataset.take(np.arange(batch_size)))[0])

    # store models and reports over time
    models = []
    reports = []

    # create optimizer and loss
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.NLLLoss()

    # train model
    training_start = time.time()
    try:
        drafts_trained = 0
        last_report = 0
        total_loss = 0

        # initialize progress bar
        bar = tqdm(total=evaluate_every)
        bar.unit = " Draft-States"

        while True:
            for i, batch in enumerate(dataset.to_batches(batch_size=batch_size)):
                # skip over test batches
                if i < num_test_batches:
                    continue

                # randomly skip batches to "shuffle" them
                if RNG.random() > 0.33:
                    continue

                # get a new batch
                games, picks = parse_batch(batch)  # parse_batch also shuffles the games within the batch
                drafts_trained += len(games)
                bar.update(len(games))

                # embed the batch
                games = embedder(games)
                games = torch.from_numpy(games).to(DEVICE)

                # predict
                pred_picks = model(games)

                # calculate loss
                picks = torch.from_numpy(picks).to(DEVICE)
                loss = loss_fn(pred_picks, picks)
                total_loss += loss.item()

                # backpropagate
                loss.backward()
                optim.zero_grad()
                optim.step()

                # evaluate every once in a while
                if drafts_trained - last_report >= evaluate_every:
                    bar.reset(evaluate_every)
                    bar.update(drafts_trained - evaluate_every - last_report)
                    last_report = drafts_trained
                    model.eval()

                    with torch.no_grad():
                        # select a random batch from the test ones
                        for test_batch, _ in zip(dataset.to_batches(batch_size=batch_size), range(RNG.integers(low=0, high=num_test_batches))):
                            pass

                        # parse the batch
                        games, picks = parse_batch(test_batch)
                        drafts_trained += len(games)

                        # embed the batch
                        games = embedder(games)
                        games = torch.from_numpy(games).to(DEVICE)

                        # predict
                        pred_picks = model(games).cpu()

                        # report
                        report = (
                            f"\nTop-10 Accuracy: {sklearn.metrics.top_k_accuracy_score(picks, pred_picks, k=10):.2%}\n"
                            f"total loss since last report: {total_loss:.5f}\n"
                            f"{(time.time() - training_start) / 60:.2f} m; Epoch {drafts_trained / 5 // dataset.count_rows():_} ; Report {len(reports)}\n"
                        )

                        models.append(np.copy(model))
                        reports.append(report)

                    model.train()
                    total_loss = 0


    finally:
        # create a directory for the model, embedder and params within the main model directory
        real_save_dir = MODEL_SAVE_DIR / f"{int(time.time())}"
        real_save_dir.mkdir(parents=True, exist_ok=True)

        # save model, embedder and params
        with open(real_save_dir / "models.dill", "w") as f:
           f.write(dumps(models))
        with open(real_save_dir / "embedder.dill", "w") as f:
            f.write(dumps(embedder))
        with open(real_save_dir / "reports.txt", "w") as f:
            f.writelines(reports)
        with open(real_save_dir / "params.json", "w") as f:
            f.write(dumps(params))
