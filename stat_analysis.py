import pickle
import time
from copy import copy

import duckdb
import pyarrow
from numpy import uint8
import click
from tqdm import tqdm

import lib

# open LoL dataset
_dataset = lib.league_of_parquet.open_dataset('./data/dataset')

# calculate maximum length of champion names
_max_champ_name_len = max([len(champion) for champion in lib.CHAMPIONS.keys()])


@click.group()
@click.option("--dataset-path", type=click.Path(dir_okay=True, file_okay=False), default="./data/dataset", help="Path to the dataset (default: ./data/dataset)")
def cli(dataset_path: str) -> None:
    """
    Statistic analysis tool for the dataset.
    """
    global _dataset
    _dataset = lib.league_of_parquet.open_dataset(dataset_path)

    print(f"Total number of matches in the dataset: {_dataset.count_rows()}")


@cli.command("winrate")
@click.argument("champion", type=click.Choice(lib.CHAMPIONS))
def cli_champ_winrate(champion: str):
    """
    Calculate the winrate of a specific champion.
    Will return blue side, red side and total winrate as well as number of games played.
    """
    bs_winrate, rs_winrate, total_winrate, total_games = champ_winrate(champion)
    print(
        f"{champion}: blue side winrate of {bs_winrate:.2%}, red side winrate of {rs_winrate:.2%}, total winrate of {total_winrate:.2%} with {total_games} matches")


def champ_winrate(champ: str) -> tuple[float, float, float, int]:
    """
    :return: tuple (blue side winrate, red side winrate, total winrate, number of games played)
    """
    # encode champion to int
    champ_int = lib.encoded_champ_id.name_to_int(champ)

    # select all games where the champ got picked
    games = duckdb.sql(f"select * from _dataset where list_contains(picks, {champ_int})")

    # count total games
    total_games = games.count("win").fetchone()[0]

    # filter out the blue side and red side picks respectively
    bs_games = games.filter(f"list_contains(picks[1:5], {champ_int})")
    rs_games = games.filter(f"list_contains(picks[6:10], {champ_int})")

    # count blue side wins/losses
    bs_wins = bs_games.filter("win").count("win").fetchone()[0]
    bs_losses = bs_games.filter("not win").count("win").fetchone()[0]

    # count red side wins/losses
    rs_wins = rs_games.filter("not win").count("win").fetchone()[0]
    rs_losses = rs_games.filter("win").count("win").fetchone()[0]

    # calculate winrates
    bs_winrate = bs_wins / (bs_wins + bs_losses)
    rs_winrate = rs_wins / (rs_wins + rs_losses)
    total_winrate = (bs_wins + rs_wins) / total_games

    return bs_winrate, rs_winrate, total_winrate, total_games


@cli.command("synergy")
@click.argument("champions", nargs=2, type=click.Choice(lib.CHAMPIONS))
def cli_synergy(champions: tuple[str, str]):
    """
    Calculate the winrate of a synergy between two champions within the same team.
    Will return the winrate of the synergy as well as the number of games played.
    """
    synergy_result = synergy(champions[0], champions[1])

    if synergy_result is None:
        print("No matches with this synergy in the dataset.")
    else:
        synergy_winrate, total_games = synergy_result
        print(
            f"{champions[0]} and {champions[1]} have a total winrate of {synergy_winrate:.2%} with {total_games} matches.")


def synergy(champ0: str, champ1: str, preselected0: pyarrow.Table = None, preselected1: pyarrow.Table = None) -> tuple[float, int] | None:
    """
    :return: tuple (winrate, number of games played)
    """
    # encode champions
    champ0_int = lib.encoded_champ_id.name_to_int(champ0)
    champ1_int = lib.encoded_champ_id.name_to_int(champ1)

    # select all games where both were picked
    if preselected0 is None:
        games = duckdb.sql(
            f"select * from _dataset where list_contains(picks, {champ0_int}) and list_contains(picks, {champ1_int})")
    else:
        assert preselected1 is not None
        games = duckdb.sql(
            f"select * from preselected0 where list_contains(picks, {champ1_int}) union select * from preselected1 where list_contains(picks, {champ0_int})")

    # only ones where both champions are on the same team
    games = games.filter(f"list_contains(picks[1:5], {champ0_int}) = list_contains(picks[1:5], {champ1_int})")
    # count all entries in the win column (losses and wins)
    total_games = games.count("win").fetchone()[0]

    # check if there are some games with this synergy
    if total_games == 0:
        return None

    # count wins and losses
    wins = games.filter(
        f"(list_contains(picks[1:5], {champ0_int}) and win) or (list_contains(picks[6:10], {champ0_int}) and not win)")
    wins = wins.count("win").fetchone()[0]
    losses = total_games - wins

    # calculate winrate
    synergy_winrate = wins / (wins + losses)

    return synergy_winrate, total_games


@cli.command("all-champ-synergies")
@click.argument("champion", type=click.Choice(lib.CHAMPIONS))
def cli_all_champ_synergies(champion: str):
    """
    Look up all synergies for a champion.
    Some synergies may not be included if there are no matches with them in the dataset.
    """
    print("Calculating synergies for all champions... This may take a while.")
    synergies = all_champ_synergies(champion)

    # sort synergies in descending order and filter out unplayed
    synergies = [(k, v) for k, v in synergies.items() if v is not None]
    sorted_synergies = sorted(synergies, key=lambda x: x[1][0], reverse=True)

    # print output
    print(f"Synergies with {champion}:")
    for alternate, (winrate, total_games) in sorted_synergies:
        print(f"{alternate:{_max_champ_name_len}} winrate of {winrate:.2%}, {total_games:5} matches")


def all_champ_synergies(champion: str, log_output: bool = True) -> dict[str, tuple[float, int] | None]:
    """
    :return: dict (alternate: (winrate, num of matches))
    """
    # initialize dictionary
    synergies = {}

    # get the list of champions
    champions = copy(lib.CHAMPIONS) if log_output is False else tqdm(copy(lib.CHAMPIONS))

    # loop all champions
    for alternate in champions:
        # get synergy winrate and store in dict (this may be None)
        synergies[alternate] = synergy(champion, alternate)

    return synergies


@cli.command("all-set-synergies")
@click.option("--output-path", type=click.Path(dir_okay=False, file_okay=True), default="./all_set_synergies.txt",
              help="Default: './all_set_synergies.txt'")
def cli_all_set_synergies(output_path: str):
    """
    Calculate all possibles synergies (every champion with every other champion) in the whole dataset.
    This will write the output to a text file.
    Some synergies may not be included if there are no matches with them in the dataset.
    Warning: slow!
    """
    all_set_synergies(output_path=output_path)


def all_set_synergies(output_path: str = None) -> dict[str, dict[str, tuple[float, int]]]:
    """
    This function returns a dictionary {champ0: {champ1: (winrate, num of matches)}}.
    """
    # initialize dictionary
    duo_wr_dict: dict[str, dict[str, tuple[float, int]]] = {}
    for champion in lib.CHAMPIONS:
        duo_wr_dict[champion]: dict[str, tuple[float, int]] = {}

    # clone champions to only compute n^2 / 2
    alternates = list(lib.CHAMPIONS.keys())

    # add progress bar
    if output_path:
        print("Preselecting champion specific tables...")
    champions = copy(alternates) if not output_path else tqdm(copy(alternates))

    # preselect champion specific tables
    champ_tables: dict[str, pyarrow.Table] = {}
    for champion in copy(champions):
        champ_tables[champion] = duckdb.sql(
            f"select * from _dataset where list_contains(picks, {lib.encoded_champ_id.name_to_int(champion)})").arrow()

    if output_path:
        print("Computing synergies...")

    # calculate all synergies within the dataset
    for champion in champions:
        for alternate in alternates:
            # get synergy winrate
            synergy_result = synergy(champion, alternate, preselected0=champ_tables[champion], preselected1=champ_tables[alternate])

            # store winrate in dict
            duo_wr_dict[champion][alternate] = synergy_result
            duo_wr_dict[alternate][champion] = synergy_result

        # remove completed champions for efficiency
        alternates.remove(champion)

    # write output to a text file if wanted
    if output_path:
        print(f"Saving synergies to {output_path}")
        with open(output_path, "w") as f:
            print(f"{_dataset.count_rows()} total number of matches in the dataset\n", file=f)
            for champion, winrates in duo_wr_dict.items():
                print("#", champion, file=f)
                print(" " * (4 + _max_champ_name_len), " win%   matches", file=f)

                # sort by winrate in descending order (and filter out unplayed synergies)
                winrates = [(k, v) for k, v in winrates.items() if v is not None]
                winrates = sorted(winrates, key=lambda x: x[1][0], reverse=True)

                # print winrate for this synergy
                for alternate, (winrate, num_matches) in winrates:
                    print(f"    {alternate:{_max_champ_name_len}} {winrate:06.2%}  {num_matches:>5}", file=f)

                print("\n", file=f)
        print("Done.")

    return duo_wr_dict


@cli.command("team-comp-wr")
@click.argument("champions", nargs=5, type=click.Choice(lib.CHAMPIONS))
def cli_team_comp_wr(champions: tuple[str, str, str, str, str]):
    """
    Calculate the winrate of a specific team comp.
    """
    # calculate team comp winrate
    team_comp_winrate = team_comp_wr(champions)

    # print output
    if team_comp_winrate is None:
        print("No games with this team comp found. :(")
    else:
        winrate, num_games = team_comp_winrate
        print(f"The team consists of: {champions[0]}, {champions[1]}, {champions[2]}, {champions[3]} and {champions[4]}.")
        print(f"This team has a winrate of {winrate:.2%} with {num_games} games played. Don't be disappointed.")


def team_comp_wr(team_comp: tuple[str, str, str, str, str]) -> tuple[float, int] | None:
    """
    Return the winrate of a team comp and number of games played, None if no games with this comp.
    """
    # encode champions
    champs: list[uint8] = [lib.encoded_champ_id.name_to_int(champ) for champ in team_comp]

    # select all games where all were picked
    games = duckdb.sql(
        f"select * from _dataset where list_contains(picks, {champs[0]}) and list_contains(picks, {champs[1]}) and list_contains(picks, {champs[2]}) and list_contains(picks, {champs[3]}) and list_contains(picks, {champs[4]})")

    # only ones where all champions are on the same team
    for champ in champs[1:]:
        games = games.filter(f"list_contains(picks[1:5], {champs[0]}) = list_contains(picks[1:5], {champ})")

    total_games = games.count("win").fetchone()[0]

    # count wins
    wins = games.filter(
        f"(list_contains(picks[1:5], {champs[0]}) and win) or (list_contains(picks[6:10], {champs[0]}) and not win)")
    wins = wins.count("win").fetchone()[0]

    # calculate losses
    losses = total_games - wins

    if total_games == 0:
        return None

    return wins / (wins + losses), total_games


if __name__ == "__main__":
    cli()
