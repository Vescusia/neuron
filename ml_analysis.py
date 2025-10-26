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
@click.option("--batch-size", type=int, default=50_000, help="Game batch size for training (default: 50,000) (reducing this will help with VRAM)")
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
@click.option("--batch-size", type=int, default=25_000, help="Game batch size for training (default: 25,000) (reducing this will help with VRAM)")
@click.option("--evaluate-every", type=int, default=1, help="Evaluate every n epochs (default: 1)")
def train_draft_model(batch_size, evaluate_every):
    """Train the draft model; models will be saved to ./analysis/draft_ml/models"""
    from analysis.draft_ml.train_model import train_model
    train_model(batch_size=batch_size, evaluate_every=evaluate_every)


@cli_group.command("draft")
@click.argument("model-path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("tier", type=click.Choice(lib.TIERS))
@click.option("--division", type=click.Choice(lib.DIVISIONS), default=lib.DIVISIONS[0])
@click.option("--report-index", type=int, default=-1, help="Index of the report state to load (default: -1)")
def draft(model_path, tier, division, report_index):
    model_path = Path(model_path)

    # encode ranked score
    ranked_score = lib.encoded_rank.to_int(tier, division)

    # open and load models
    model = ml_lib.load_model(model_path / "models.dill", state_dict_index=report_index)
    model.eval()
    click.echo(f"Comp models loaded from {model_path}")

    # open and load embedder
    embedder = ml_lib.Embedder.load(model_path / "embedder.dill")
    click.echo(f"Embedder loaded from {model_path}")

    # prompt the user for blue side or red side
    bs_or_rs = ""
    while bs_or_rs not in ["bs", "rs"]:
        bs_or_rs = click.prompt("Are you Blue Side or Red Side (bs/rs)")
    bs_or_rs = 1 if bs_or_rs == "bs" else 0

    # define bans as 10 no pick/ban
    bans = [0] * 10

    # define picks as 10 no pick
    picks = [0] * 10

    def predict_pick(n=10):
        # embed draft state
        draft_state = picks + bans + [ranked_score] + [bs_or_rs]
        draft_state = embedder(np.array([draft_state]))

        # score picks
        with torch.no_grad():
            pred_picks = model(torch.from_numpy(draft_state))
            pred_picks = pred_picks.numpy()[0]

        # get the indices of the n most probable champions
        top = np.argsort(-pred_picks)[:n]

        # decode the champions to names
        champions = [lib.encoded_champ_id.int_to_name(champ + 1) for champ in top]   # +1 because the model does not predict no pick
        champions = np.array(champions)

        # and get their prediction probabilities (and invert the log of LogSoftMax)
        probabilities = np.exp(pred_picks[top])

        click.echo(" ".join([f"{champ}: {probab:.2%}" for champ, probab in zip(champions, probabilities)]))

    def prompt_pick(pick_idx=None, ban_idx=None):
        while True:
            # prompt user for the pick
            pick = click.prompt("Pick a Champion")

            if pick not in lib.CHAMPIONS.keys():
                click.echo(f"{pick} is not known.")
                # echo all champions that have the same initial character
                click.echo(f"Similar Champions: {[champ for champ in lib.CHAMPIONS.keys() if champ.startswith(pick[0].upper())]}")
            else:
                break

        # encode pick such that the model understands it
        encoded = lib.encoded_champ_id.name_to_int(pick)

        # write to picks/bans
        if pick_idx is not None:
            picks[pick_idx] = int(encoded)
        elif ban_idx is not None:
            bans[ban_idx] = int(encoded)
        else:
            exit(-1)

    click.echo("1st Bans (bs):")
    [prompt_pick(ban_idx=idx) for idx in range(3)]
    click.echo("1st Bans (rs):")
    [prompt_pick(ban_idx=idx) for idx in range(5, 8)]

    click.echo("1st pick:")
    if bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=0)

    click.echo("2nd/3rd pick:")
    if not bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=5)
    if not bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=6)

    click.echo("4th/5th pick:")
    if bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=1)
    if bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=2)

    click.echo("6th pick:")
    if not bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=7)

    click.echo("2nd Bans (bs):")
    [prompt_pick(ban_idx=idx) for idx in range(3, 5)]
    click.echo("2nd Bans (rs):")
    [prompt_pick(ban_idx=idx) for idx in range(8, 10)]

    click.echo("7th pick:")
    if not bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=8)

    click.echo("8th/9th pick:")
    if bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=3)
    if bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=4)

    click.echo("last pick:")
    if not bs_or_rs:
        predict_pick()
    prompt_pick(pick_idx=9)


if __name__ == "__main__":
    cli_group()
