import click


@click.group()
@click.option("--dataset-path", type=click.Path(dir_okay=True, file_okay=False), default="./data/dataset", help="Path to the dataset (default: ./data/dataset)")
def cli_group(dataset_path):
    """ML analysis tool for neuron datasets"""
    pass

@cli_group.command("train_comp_model")
@click.option("--batch-size", type=int, default=50_000, help="Game batch size for training (default: 50,000)")
@click.option("--evaluate-every", type=int, default=10_000_000, help="Evaluate every n games (default: 10,000,000)")
def train_comp_model(batch_size, evaluate_every):
    """Train the comp model; models will be saved to ./analysis/comp_ml/models"""
    from analysis.comp_ml.train_model import train_model
    train_model(batch_size=batch_size, evaluate_every=evaluate_every)

@cli_group.command("train_draft_model")
@click.option("--batch-size", type=int, default=16_000, help="Game batch size for training (default: 16,000)")
@click.option("--evaluate-every", type=int, default=10_000_000, help="Evaluate every n draft-states (default: 10,000,000)")
def train_comp_model(batch_size, evaluate_every):
    """Train the draft model; models will be saved to ./analysis/draft_ml/models"""
    from analysis.comp_ml.train_model import train_model
    train_model(batch_size=batch_size, evaluate_every=evaluate_every)


if __name__ == "__main__":
    cli_group()
