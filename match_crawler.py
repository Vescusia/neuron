from pathlib import Path

import click
import riotwatcher as rw

import lib
import data_snake


@click.command()
@click.argument("api-key")
@click.option("-c", "--continent", default="ALL", help="Continents to Scan (default: ALL)", type=click.Choice(lib.CONTINENTS + ["ALL"]))
@click.option("--data-path", default="./data", type=click.Path(dir_okay=True, file_okay=False), help="Path to the directory within which the match data will be saved. "
                                                                                                     "The content of this directory should not be manually managed. (Default: './data')")
def main(api_key: str, continent: str, data_path: str) -> None:
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
    data_snake.gather(continents, match_db_path, sum_db_path, matches_path, dataset_path, lolwatcher)


if __name__ == "__main__":
    main()
