from pathlib import Path

from neuron import main

import click
from cassiopeia.data import Continent
import cassiopeia as cass


@click.command()
@click.argument("api-key")
@click.option("-c", "--continent", default="ALL", help="Continents to Scan (default: ALL)", type=click.Choice([c.value for c in Continent] + ["ALL"]))
@click.option("-mdb", "--match-db", default="./db/match_db", type=click.Path(dir_okay=True, file_okay=False), help="Path to Match Database (Directory!) (default: ./db/match_db")
@click.option("-sdb", "--sum-db", default="./db/sum_db", type=click.Path(dir_okay=True, file_okay=False), help="Path to Summoner Database (Directory!) (default: ./db/match_db")
@click.option("--syn-db", default="./db/synergies/syn_db", type=click.Path(dir_okay=True, file_okay=False), help="Path to Synergy Database (Directory!) (default: ./db/synergies/syn_db")
@click.option("--matchup-db", default="./db/synergies/matchup_db", type=click.Path(dir_okay=True, file_okay=False), help="Path to Matchup Database (Directory!) (default: ./db/synergies/matchup_db")
@click.option("--matches", default="./db/matches", type=click.Path(dir_okay=True, file_okay=False), help="Path to the Directory in which the Match Json will be saved (Directory!) (default: ./db/matches")
def cli(api_key: str, continent: str, match_db: str, sum_db: str, syn_db: str, matchup_db: str, matches: str) -> None:
    # parse continents
    continents = []
    match continent:
        case "ALL":
            [continents.append(c) for c in Continent]
        case i:
            continents.append(Continent(i))

    # test key validity
    cass.set_riot_api_key(api_key)
    try:
        chal = cass.get_challenger_league(cass.Queue.ranked_solo_fives, "EUW")
        _ = chal[0]
    except Exception as e:
        click.echo(e)
        return

    main(api_key, continents, match_db, sum_db, syn_db, matchup_db, Path(matches))


if __name__ == "__main__":
    cli()
