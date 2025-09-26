from pathlib import Path

import click
import torch
import numpy as np
import dill

import lib


@click.group()
@click.option("--dataset-path", type=click.Path(dir_okay=True, file_okay=False), default="./data/dataset", help="Path to the dataset (default: ./data/dataset)")
def cli_group(dataset_path):
    """ML analysis tool for neuron datasets"""
    pass


@cli_group.command("train-comp-model")
@click.option("--batch-size", type=int, default=50_000, help="Game batch size for training (default: 50,000)")
@click.option("--evaluate-every", type=int, default=10_000_000, help="Evaluate every n games (default: 10,000,000)")
def train_comp_model(batch_size, evaluate_every):
    """Train the comp model; models will be saved to ./analysis/comp_ml/models"""
    from analysis.comp_ml.train_model import train_model
    train_model(batch_size=batch_size, evaluate_every=evaluate_every)


@cli_group.command("comp_model")
@click.argument("model-path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("report-index", type=int, default=0)
@click.argument("champions", nargs=10, type=click.Choice(lib.CHAMPIONS))
@click.argument("patch", type=click.Choice(lib.ALL_PATCHES))
@click.argument("tier", type=click.Choice(lib.TIERS))
@click.option("--division", type=click.Choice(lib.DIVISIONS), default=lib.DIVISIONS[0])
def use_comp_model(model_path, report_index, champions, patch, tier, division):
    model_path = Path(model_path)

    # open and load models
    with open(model_path / "models.dill", "rb") as f:
        models = dill.load(f)
    click.echo(f"Comp models loaded from {model_path}")

    # open and load embedder
    with open(model_path / "embedder.dill", "rb") as f:
        embedder = dill.load(f)
    click.echo(f"Embedder loaded from {model_path}")

    # encode champions, patch and rank
    champions = [lib.encoded_champ_id.name_to_int(champ) for champ in champions]
    game = np.array(champions + [lib.encoded_patch.to_int(patch), lib.encoded_rank.to_int(tier, division)], dtype=np.uint16).reshape((1, -1))

    # embed game
    game = embedder(game)
    game = torch.from_numpy(game)

    # select model
    model = models[report_index]
    model.eval()

    # predict
    with torch.no_grad():
        prediction = model(game)

    click.echo(f"BS has a probable winrate of {prediction.item():.2%}")


@cli_group.command("train-draft-model")
@click.option("--batch-size", type=int, default=16_000, help="Game batch size for training (default: 16,000)")
@click.option("--evaluate-every", type=int, default=10_000_000, help="Evaluate every n draft-states (default: 10,000,000)")
def train_draft_model(batch_size, evaluate_every):
    """Train the draft model; models will be saved to ./analysis/draft_ml/models"""
    from analysis.draft_ml.train_model import train_model
    train_model(batch_size=batch_size, evaluate_every=evaluate_every)


if __name__ == "__main__":
    cli_group()
