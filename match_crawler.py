from pathlib import Path

import click
import riotwatcher as rw

import lib
import data_snake


@click.command()
@click.argument("api-key")
@click.option("--data-path", default="./data", type=click.Path(dir_okay=True, file_okay=False), help="Path to the directory within which the match data will be saved. "
                                                                                                     "The content of this directory should not be manually managed. (Default: './data')")
@click.option("--match-map-size", default=1_500_000_000, type=click.INT, help="The size of the Match LMDB map in Bytes (default: 1_500_000_000)")
@click.option("--sum-map-size", default=4_000_000_000, type=click.INT, help="The size of the Summoner LMDB map in Bytes (default: 4_000_000_000)")
@click.option("-c", "--continent", default="ALL", help="Continents to Scan/Migrate (default: ALL)", type=click.Choice(lib.CONTINENTS + ["ALL"]))
def gather(api_key: str, data_path: str, match_map_size: int, sum_map_size: int, continent) -> None:
    """
    Gather ranked Matches and Summoners from multiple Continents.
    """
    lolwatcher = rw.LolWatcher(api_key)

    # parse continents
    continents = []
    match continent:
        case "ALL":
            [continents.append(c) for c in lib.CONTINENTS]
        case i:
            continents.append(i)

    # construct the paths for the subdirectories within the data directory
    data_path = Path(data_path)
    match_db_path = data_path / "match_db"
    sum_db_path = data_path / "sum_db"
    matches_path = data_path / "matches"
    dataset_path = data_path / "dataset"

    # call the main match gathering function
    data_snake.gather(
        continents, match_db_path, sum_db_path, matches_path, dataset_path, lolwatcher, match_map_size, sum_map_size)


if __name__ == "__main__":
    gather()
