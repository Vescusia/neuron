from pathlib import Path

import click
import torch
import numpy as np

import lib
from analysis import ml_lib


_DATASET_PATH: str | None = None


@click.group()
@click.option("--dataset-path", type=click.Path(dir_okay=True, file_okay=False), default="./data/dataset", help="Path to the dataset (default: ./data/dataset)")
def cli_group(dataset_path):
    """ML analysis tool for neuron datasets"""
    global _DATASET_PATH
    _DATASET_PATH = dataset_path


@cli_group.command("train-comp-model")
@click.option("--batch-size", type=int, default=50_000, help="Game batch size for training (default: 50,000)")
@click.option("--evaluate-every", type=int, default=20_000_000, help="Evaluate every n games (default: 20,000,000)")
def train_comp_model(batch_size, evaluate_every):
    """Train the comp model; models will be saved to ./analysis/comp_ml/models"""
    from analysis.comp_ml.train_model import train_model
    train_model(dataset_path=_DATASET_PATH, batch_size=batch_size, evaluate_every=evaluate_every)


@cli_group.command("comp-model")
@click.argument("model-path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.option("--report-index", type=int, default=-1, help="Index of the report state to load (default: -1)")
@click.argument("champions", nargs=10, type=click.Choice(lib.CHAMPIONS))
@click.argument("tier", type=click.Choice(lib.TIERS))
@click.option("--division", type=click.Choice(lib.DIVISIONS), default=lib.DIVISIONS[0])
def use_comp_model(model_path, report_index, champions, tier, division):
    model_path = Path(model_path)

    # open and load models
    model = ml_lib.load_model(model_path / "models.dill", state_dict_index=report_index)
    model.eval()
    click.echo(f"Comp models loaded from {model_path}")

    # open and load embedder
    embedder = ml_lib.Embedder.load(model_path / "embedder.dill")
    click.echo(f"Embedder loaded from {model_path}")

    # encode champions, patch and rank
    champions = [lib.encoded_champ_id.name_to_int(champ) for champ in champions]
    game = np.array(champions + [lib.encoded_rank.to_int(tier, division)], dtype=np.uint16).reshape((1, -1))

    # embed game
    game = embedder(game)
    game = torch.from_numpy(game)

    # predict
    with torch.no_grad():
        prediction = model(game)

    click.echo(f"BS has a probable winrate of {prediction.item():.2%}")


@cli_group.command("train-draft-model")
@click.option("--batch-size", type=int, default=20_000, help="Game batch size for training (default: 20,000)")
@click.option("--evaluate-every", type=int, default=1, help="Evaluate every n epochs (default: 1)")
def train_draft_model(batch_size, evaluate_every):
    """Train the draft model; models will be saved to ./analysis/draft_ml/models"""
    from analysis.draft_ml.train_model import train_model
    train_model(batch_size=batch_size, evaluate_every=evaluate_every)


if __name__ == "__main__":
    cli_group()
