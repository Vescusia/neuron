from pathlib import Path
import time
import json

import sklearn
import torch
import numpy as np
from tqdm import tqdm
from pyarrow import Table as PyArrowTable

import lib.league_of_parquet as lop
from .model import ResNet20, DraftEmbedder
from analysis import ml_lib

# define dataset path
_DATASET_PATH = Path("./data/dataset")

# define save directory for the model
_MODEL_SAVE_DIR = Path("./analysis/draft_ml/models")

# define parameters for model and lr scheduler
_PARAMS = {
    "num_champions": 171,
    "model": {
        "width": 256,
        "bottleneck": 8,
        "dropout": 0.5,
        "blocks_individual": 4,
        "blocks_pre_win": 5,
        "blocks_pre_bans": 5,
        "blocks_pre_rank": 5,
        "blocks_post_rank": 5
    },
    "lr": {
        "initial_lr": 0.001,  # should be roughly <1/25 of max_lr
        "max_lr": 0.01,
        "min_lr": 0.00001,
        "one_cycle_epochs": 40,
    }
}


class DraftDataset(torch.utils.data.Dataset):
    def __init__(self, num_champions: int, table: PyArrowTable):
        self.table = table
        self.embedder = DraftEmbedder(num_champions)
        self.rng = np.random.default_rng(int(time.time()))

    def __getitems__(self, idxs: list[int], _fit=False):
        # get the batch of games from the dataset
        batch = self.table.take(idxs)

        # load wins
        wins = batch['win'].to_numpy().astype(dtype=np.uint16)

        # load ranked scores
        ranked_scores = batch['ranked_score'].to_numpy().astype(dtype=np.uint16)

        # load picks
        picks = np.array(batch['picks'].to_numpy().tolist(), dtype=np.uint16)
        # shuffle picks (per game, per side) to prevent top lane always being first pick (and so on)
        picks = picks.reshape((-1, 5))
        self.rng.shuffle(picks, axis=1)
        picks = picks.reshape((-1, 10))

        # load bans
        bans = np.array(batch['bans'].to_numpy().tolist(), dtype=np.uint16)

        # repeat games by 5, as there are 5 predictable "winning" picks per game
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
        blue_targets = blue_targets[np.arange(len(blue_targets)), np.tile(np.arange(5), len(blue_targets) // 5)]

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

        # fit the embedder if wanted
        if _fit:
            self.embedder.fit(games)

        # embed the games
        embedded_games = self.embedder(games)

        return embedded_games, all_targets

    def fit(self, idxs):
        """
        Fit the scaler on ``idxs``.
        """
        self.__getitems__(idxs, _fit=True)

    def __getitem__(self, idx: int):
        return self.__getitems__([idx])

    def __len__(self):
        return len(self.table)

    @staticmethod
    def collate_fn(batch):
        games, targets = batch
        return torch.from_numpy(games).to(torch.float32), torch.from_numpy(targets).to(torch.int64)


def train_model(batch_size: int, evaluate_every: int):
    # check for CUDA availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device("cuda")
    else:
        print("No GPU available.")
        device = torch.device("cpu")

    # load parquet dataset
    print(f"Opening dataset from {_DATASET_PATH}")
    start = time.time()
    dataset = lop.open_dataset(_DATASET_PATH)
    table = dataset.to_table()
    print(f"Dataset loaded to memory in {time.time() - start:.2f} s")

    # convert parquet dataset to torch dataset
    dataset = DraftDataset(_PARAMS["num_champions"], table)
    print(f"Dataset contains {len(dataset):_} games")

    # fit scaler of parquet dataset
    dataset.fit(np.random.randint(0, len(dataset), batch_size))

    # create subset for train and test data
    test_indices = np.random.randint(0, len(dataset), len(dataset) // 10)
    test_data = torch.utils.data.Subset(dataset, test_indices)
    train_data = torch.utils.data.Subset(dataset, np.delete(np.arange(len(dataset)), test_indices))

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=DraftDataset.collate_fn, num_workers=6, pin_memory=torch.cuda.is_available())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=DraftDataset.collate_fn)
    test_loader_batches = iter(test_loader)

    # create model
    model = ResNet20(num_champions=_PARAMS["num_champions"], **_PARAMS["model"]).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):_} parameters")

    # store models and reports over time
    models = []
    reports = [
        f"{model}\n",
        f"{sum(p.numel() for p in model.parameters()):_} parameters\n"
    ]

    # create optimizer
    optim = torch.optim.Adam(model.parameters())

    # create lr scheduler
    lr_scheduler = ml_lib.lr_one_cycle_till_cyclic(
        optim,
        **_PARAMS["lr"],
        steps_per_epoch=len(train_loader)
    )

    # define loss function
    criterion = torch.nn.NLLLoss()
    torch.autograd.set_detect_anomaly(True)

    # train model
    training_start = time.time()
    try:
        epoch = 0
        total_loss = 0

        while True:
            for i, (draft_state, winning_pick) in enumerate(tqdm(train_loader)):
                draft_state, winning_pick = draft_state.to(device), winning_pick.to(device)

                # predict
                pred_picks = model(draft_state)

                # calculate loss
                loss = criterion(pred_picks, winning_pick)
                total_loss += loss.item()

                # backpropagate
                optim.zero_grad()
                loss.backward()
                optim.step()

                # step the learning rate
                lr_scheduler.step()

            # increment epoch after having seen the whole training data
            epoch += 1

            # evaluate every n epochs
            if epoch % evaluate_every == 0:
                model.eval()
                with torch.no_grad():
                    # get a new batch from the test loader
                    try:
                        batch = next(test_loader_batches)
                    except StopIteration:
                        test_loader_batches = iter(test_loader)
                        batch = next(test_loader_batches)

                    # get draft state and target out of batch
                    draft_state, winning_pick = batch

                    # predict
                    pred_picks = model(draft_state.to(device)).cpu().numpy()

                    # calculate average maximum confidence
                    max_confs = np.amax(pred_picks, axis=1)
                    avg_max_confidence = np.exp(np.mean(max_confs))

                    # report
                    report = (
                        f"\nTop-10 Accuracy: {sklearn.metrics.top_k_accuracy_score(winning_pick, pred_picks, k=10):.2%}"
                        f"\naverage maximum confidence: {avg_max_confidence:.2%}"
                        f"\ntotal loss since last report: {total_loss:.5f}"
                        f"\ncurrent lr: {lr_scheduler.get_last_lr()[0]:.5f}"
                        f"\n{(time.time() - training_start) / 60:.2f} m; Epoch {epoch} ; Report {epoch // evaluate_every}"
                    )
                    print(report)

                    # save report and current model state
                    models.append(model.state_dict())
                    reports.append(report)

                model.train()
                total_loss = 0

    finally:
        # create a directory for the model, embedder and params within the main model directory
        real_save_dir = _MODEL_SAVE_DIR / f"{int(time.time())}"
        real_save_dir.mkdir(parents=True, exist_ok=True)

        # save model
        ml_lib.save_model(model.cpu(), {"num_champions": _PARAMS["num_champions"], **_PARAMS["model"]}, models, real_save_dir / "models.dill")
        # save embedder
        dataset.embedder.save(real_save_dir / "embedder.dill", {"num_champions": _PARAMS["num_champions"]})
        # save params
        with open(real_save_dir / "params.json", "w") as f:
            json.dump(_PARAMS, f)
        # save reports
        with open(real_save_dir / "reports.txt", "w") as f:
            f.writelines(reports)

        print(f"Saved model to {real_save_dir}.")
