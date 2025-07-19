from calendar import leapdays
from pathlib import Path

import click
import riotwatcher as rw

import lib
from data_snake import main


@click.command()
@click.argument("api-key")
@click.option("-c", "--continent", default="ALL", help="Continents to Scan (default: ALL)", type=click.Choice(lib.CONTINENTS + ["ALL"]))
@click.option("-mdb", "--match-db", default="./data/match_db", type=click.Path(dir_okay=True, file_okay=False), help="Path to Match Database (Directory!) (default: ./data/match_db")
@click.option("-sdb", "--sum-db", default="./data/sum_db", type=click.Path(dir_okay=True, file_okay=False), help="Path to Summoner Database (Directory!) (default: ./data/match_db")
@click.option("--matches", default="./data/matches", type=click.Path(dir_okay=True, file_okay=False), help="Path to the Directory in which the Match Json will be saved (Directory!) (default: ./data/matches")
def cli(api_key: str, continent: str, match_db: str, sum_db: str, matches: str) -> None:
    lolwatcher = rw.LolWatcher(api_key)

    # parse continents
    continents = []
    match continent:
        case "ALL":
            [continents.append(c) for c in lib.CONTINENTS]
        case i:
            continents.append(i)

    # test key validity
    _ = lolwatcher.league.masters_by_queue("euw1", "RANKED_SOLO_5x5")

    main(api_key, continents, match_db, sum_db, Path(matches), lolwatcher)


if __name__ == "__main__":
    cli()
