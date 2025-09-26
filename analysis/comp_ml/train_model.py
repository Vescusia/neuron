import time
from pathlib import Path
import json

from tqdm import tqdm
import torch
import sklearn
import numpy as np
import dill

from .model import Embedder, ResNet60
import lib.league_of_parquet as lop

# load dataset
DATASET_PATH = Path("./data/dataset")
print(f"Opening Dataset from {DATASET_PATH}")
DATASET = lop.open_dataset(DATASET_PATH)

# define save directory for the model
MODEL_SAVE_DIR = Path("./analysis/comp_ml/models")

# check for CUDA availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    DEVICE = torch.device("cuda")
else:
    print("No GPU available.")
    DEVICE = torch.device("cpu")


def read_dataset():
    # convert pyarrow dataset to row-based pandas dataframe
    dataset = DATASET.to_table()

    # load wins into a numpy array
    wins = dataset["win"].to_numpy().astype(dtype=np.float32)

    # load ranked scores into a numpy array and reshape to 2d
    ranked_scores = dataset["ranked_score"].to_numpy()
    ranked_scores = ranked_scores.reshape(-1, 1)

    # load patches
    patches = dataset["patch"].to_numpy()
    patches = patches.reshape(-1, 1)

    # load picks into a numpy object (not array!!) which then has to be converted to python and back to an array
    # (the only way D:)
    picks = dataset["picks"].to_numpy()
    picks = np.array(picks.tolist(), dtype=np.uint16)  # uint16 prevents overflows

    # concatenate all columns into games
    games = np.concatenate((picks, patches, ranked_scores), axis=1)

    return games, wins


def train_model(batch_size=50_000, evaluate_every=10_000_000, save_all_models=True):
    # read and split dataset
    start = time.time()
    train_games, train_wins = read_dataset()
    train_games, test_games, train_wins, test_wins = sklearn.model_selection.train_test_split(train_games, train_wins, test_size=0.10)
    print(f"Loaded Dataset in {(time.time() - start):.1f} s")

    # initialize model
    params = {"num_champions": 171, "base_width": 256, "bottleneck": 5, "dropout": 0.0, "pre_rank_blocks": 2, "post_rank_blocks": 2}
    model = ResNet60(**params).to(DEVICE)
    print(f"Model has {sum(p.numel() for p in model.parameters()):_} parameters")
    # initialize embedder
    embedder = Embedder(params["num_champions"])
    embedder.fit(test_games[0:batch_size])

    # initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    # keep track of confidence reports and model states for saving the optimum model
    models = []
    reports = []

    # train loop
    start = time.time()
    total_loss = 0.0
    num_matches_seen_since_report = 0
    try:
        for epoch in range(1024):
            # split total dataset into batches
            batch_range = tqdm(range(0, len(train_games), batch_size))
            batch_range.set_description(f"EPOCH {epoch + 1}")

            # train model for this epoch
            for i in batch_range:
                num_matches_seen_since_report += batch_size

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

                if num_matches_seen_since_report >= evaluate_every:
                    model.eval()

                    # evaluate model
                    with torch.no_grad():
                        # get a batch of random games
                        games, wins = test_games[0:batch_size], test_wins[0:batch_size]
                        games = embedder(games)
                        games = torch.from_numpy(games).to(DEVICE)

                        # predict win/lose
                        predicted_wins = torch.flatten(model(games))
                        predicted_wins = predicted_wins.cpu().numpy()

                        # select the high-confidence predictions
                        high_conf_pred = np.concatenate((predicted_wins[predicted_wins > 0.60], predicted_wins[predicted_wins < 0.4]))
                        high_conf_wins = np.concatenate((wins[predicted_wins > 0.60], wins[predicted_wins < 0.4]))

                        # score the high-confidence predictions
                        high_conf_accuracy = sklearn.metrics.accuracy_score(high_conf_wins, np.round(high_conf_pred))
                        undecided = (len(predicted_wins) - len(high_conf_pred)) / len(predicted_wins)

                        # build classification report
                        report = (
                            f"\ngeneral accuracy: {sklearn.metrics.accuracy_score(wins, np.round(predicted_wins)):.2%}"
                            f"\nhigh confidence prediction accuracy: {high_conf_accuracy:.2%} with {undecided:.2%} undecided"
                            f"\nloss since last report: {total_loss:.5f}"
                            f"\nrough alpha: {model.get_parameter('res_blocks_post_rank.1.alpha').item():.5f}"
                            f"\n{(time.time() - start) / 60:.1f} m; Epoch {epoch + 1}; Report {len(reports)}"
                                  )
                        print(report)

                        # save report and current model
                        reports.append(report)
                        if save_all_models:
                            models.append(model.cpu())
                        else:
                            models[0] = model.cpu()

                    num_matches_seen_since_report = 0
                    total_loss = 0.0
                    model.train()

            # reshuffle train and test data (separately!) every epoch
            train_games, train_wins = sklearn.utils.shuffle(train_games, train_wins)
            test_games, test_wins = sklearn.utils.shuffle(test_games, test_wins)

    finally:
        # create a directory for the model, embedder and params within the main model directory
        real_save_dir = MODEL_SAVE_DIR / f"{int(time.time())}"
        real_save_dir.mkdir(parents=True, exist_ok=True)

        # save model, embedder and params
        with open(real_save_dir / "models.dill", "wb") as f:
            dill.dump(models, f, recurse=True)
        with open(real_save_dir / "reports.txt", "w") as f:
            f.writelines(reports)
        with open(real_save_dir / "embedder.dill", "wb") as f:
            dill.dump(embedder, f, recurse=True)
        with open(real_save_dir / "params.json", "w") as f:
            json.dump(params, f)


if __name__ == "__main__":
    train_model()
