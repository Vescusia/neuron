from pathlib import Path

import click
import riotwatcher as rw

import lib
import data_snake


_continents = []


@click.group()
@click.option("-c", "--continent", default="ALL", help="Continents to Scan/Migrate (default: ALL)", type=click.Choice(lib.CONTINENTS + ["ALL"]))
def cli(continent) -> None:
    global _continents

    # parse continents
    match continent:
        case "ALL":
            [_continents.append(c) for c in lib.CONTINENTS]
        case i:
            _continents.append(i)


@cli.command("gather")
@click.argument("api-key")
@click.option("--data-path", default="./data", type=click.Path(dir_okay=True, file_okay=False), help="Path to the directory within which the match data will be saved. "
                                                                                                     "The content of this directory should not be manually managed. (Default: './data')")
@click.option("--match-map-size", default=1_000_000_000, type=click.INT, help="The size of the Match LMDB map in Bytes (default: 1_000_000_000)")
@click.option("--sum-map-size", default=2_250_000_000, type=click.INT, help="The size of the Summoner LMDB map in Bytes (default: 2_250_000_000)")
def gather(api_key: str, data_path: str, match_map_size: int, sum_map_size: int) -> None:
    """
    Gather ranked Matches and Summoners from multiple Continents.
    """

    lolwatcher = rw.LolWatcher(api_key)

    # construct the paths for the subdirectories within the data directory
    data_path = Path(data_path)
    match_db_path = data_path / "match_db"
    sum_db_path = data_path / "sum_db"
    matches_path = data_path / "matches"
    dataset_path = data_path / "dataset"

    # call the main match gathering function
    data_snake.gather(
        _continents, match_db_path, sum_db_path, matches_path, dataset_path, lolwatcher, match_map_size, sum_map_size)


@cli.command("migrate-lmdb")
@click.argument("source-data-path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("source-match-map-size", type=click.INT)
@click.argument("source-sum-map-size", type=click.INT)
@click.argument("target-data-path", type=click.Path(dir_okay=True, file_okay=False, exists=False))
@click.argument("target-match-map-size", type=click.INT)
@click.argument("target-sum-map-size", type=click.INT)
def migrate(
        source_data_path: str,
        source_match_map_size: int,
        source_sum_map_size: int,
        target_data_path: str,
        target_match_map_size: int,
        target_sum_map_size: int,
) -> None:
    """
    Clone the MatchDB and SumDB LMDB Databases to a new location, with a different LMDB map size.
    This can become necessary when the LMDB map size of a Database is too small and it overflows.

    source-data-path: Path to the data directory *from* which will be copied.
    This directory should contain both the 'match_db/' and 'sum_db/' subdirectories.
    By default, _gather_ will save it's LMDB databases in './data/'

    source-match-map-size: The size of the source Match LMDB map in Bytes.

    source-sum-map-size: The size of the source Summoner LMDB map in Bytes.


    target-data-path: Path to the data directory *to* which will be copied. Should be empty.

    target-match-map-size: The size of the target Match LMDB map in Bytes.

    target-sum-map-size: The size of the target Summoner LMDB map in Bytes.
    """
    import lmdb

    from data_snake.continent_db import ContinentDB

    # convert string paths to actual Paths
    source_data_path = Path(source_data_path)
    target_data_path = Path(target_data_path)

    # open source match db environment
    source_match_db_path = source_data_path / "match_db"
    if not source_match_db_path.exists():
        click.confirm(f"Source MatchDB directory '{source_match_db_path}' does not exist. Should MatchDB be skipped? (will create empty MatchDB at target and source)", abort=True)
    source_match_env = lmdb.open(str(source_match_db_path), map_size=source_match_map_size, max_dbs=len(_continents))

    # open source summoner db environment
    source_sum_db_path = source_data_path / "sum_db"
    if not source_sum_db_path.exists():
        click.confirm(f"Source SumDB directory '{source_sum_db_path}' does not exist. Should SumDB be skipped? (will create empty SumDB at target and source)", abort=True)
    source_sum_env = lmdb.open(str(source_sum_db_path), map_size=source_sum_map_size, max_dbs=len(_continents))

    # create target data directory
    target_data_path.mkdir(exist_ok=True)

    # open target match db environment
    target_match_path = target_data_path / "match_db"
    target_match_path.mkdir(exist_ok=False)
    target_match_env = lmdb.open(str(target_match_path), map_size=target_match_map_size, max_dbs=len(_continents))

    # open target summoner db environment
    target_sum_path = target_data_path / "sum_db"
    target_sum_path.mkdir(exist_ok=False)
    target_sum_env = lmdb.open(str(target_sum_path), map_size=target_sum_map_size, max_dbs=len(_continents))

    # migrate each continent
    for continent in _continents:
        click.echo(f"\nCloning continent {continent}")

        # define match dbs
        source_match_db = ContinentDB(source_match_env, continent)
        target_match_db = ContinentDB(target_match_env, continent)
        # migrate match database
        sourced, written = source_match_db.clone_to(target_match_db)
        click.echo(f"MatchDB: {written:_} matches cloned. (Total: {sourced:_}, {sourced - written} duplicates)")
        assert written == target_match_db.count()
        corrupted = sourced != written

        # define summoner dbs
        source_sum_db = ContinentDB(source_sum_env, continent)
        target_sum_db = ContinentDB(target_sum_env, continent)
        # migrate summoner database
        sourced, written = source_sum_db.clone_to(target_sum_db)
        click.echo(f"SumDB: {written:_} summoners cloned. (Total: {sourced:_}, {sourced - written} duplicates)")
        assert written == target_sum_db.count()
        corrupted = corrupted or sourced != written

        # check for corruption
        if corrupted:
            click.confirm(
                    f"Key duplicates in source DB detected for {continent}. "
                    f"This is usually a symptom of corruption. "
                    f"Corrupted key/value pairs will not be migrated. Do you want to continue?", abort=True)


if __name__ == "__main__":
    cli()
