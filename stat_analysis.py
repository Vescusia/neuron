import duckdb
import pickle
import click
from numpy import uint8

import lib


# open LoL dataset
_dataset = lib.league_of_parquet.open_dataset('./data/dataset')


def champ_winrate(champ: str) -> (float, float, float, int):
    """
    :return: tuple (blue side winrate, red side winrate, total winrate, num of matches)
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

    return bs_wins / (bs_wins + bs_losses), rs_wins / (rs_wins + rs_losses), (bs_wins + rs_wins) / total_games, total_games


def synergy(champ0: str, champ1: str) -> (float, int):
    # encode champions
    champ0_int = lib.encoded_champ_id.name_to_int(champ0)
    champ1_int = lib.encoded_champ_id.name_to_int(champ1)

    # select all games where both were picked
    games = duckdb.sql(f"select * from _dataset where list_contains(picks, {champ0_int}) and list_contains(picks, {champ1_int})")
    # only ones where both champions are on the same team
    games = games.filter(f"list_contains(picks[1:5], {champ0_int}) = list_contains(picks[1:5], {champ1_int})")
    # count all entries in win column (looses and wins)
    total_games = games.count("win").fetchone()[0]

    # count wins
    wins = games.filter(f"(list_contains(picks[1:5], {champ0_int}) and win) or (list_contains(picks[6:10], {champ0_int}) and not win)")
    wins = wins.count("win").fetchone()[0]

    # calculate losses
    losses = total_games - wins

    print(f"{champ0} + {champ1}: total winrate {wins / (wins + losses):.2%}, num of matches: {total_games}")
    return wins / (wins + losses), total_games

def champ_duo_winrate(champion: str, ascending: bool = False) -> {str: [(str, float, int)]}:
    """
    This function returns a dictionary for the given champion and a list of all synergy combinations for each champion.
    """
    # initialize dictionary
    duo_wr_dict = {champion: []}

    # loop all champions
    for alternate in lib.CHAMPIONS:
        if alternate == champion:
            continue
        # get synergy winrate and save
        syn_wr, num_matches = synergy(champion, alternate)
        duo_wr_dict[alternate] = (syn_wr, num_matches)

    return duo_wr_dict

def all_duo_winrates(ascending: bool = False) -> dict[str: [tuple[str, float, int]]]:
    """
    This function returns a dictionary of all champions and a list of all synergy combinations for each champion.
    """
    # initialize dictionary
    duo_wr_dict: dict[str, list[tuple[str, float, int]]] = {}
    champions_temp: list[str] = list(lib.CHAMPIONS.keys())
    for champion in lib.CHAMPIONS:
        duo_wr_dict[champion]: dict[str, tuple[float, int]] = {}

    # copy champion names
    champion_names: list[str] = list(lib.CHAMPIONS.keys())

    # go through all combinations and save them
    for champion in lib.CHAMPIONS:
        champion_int = lib.encoded_champ_id.name_to_int(champion)
        games = duckdb.sql(f"select * from _dataset where list_contains(picks, {champion_int})")

        for alternate in champion_names:
            # get synergy winrate
            syn_wr, num_matches = synergy(champion, alternate)

            # save winrate
            duo_wr_dict[champion][alternate] = (syn_wr, num_matches)
            duo_wr_dict[alternate][champion] = (syn_wr, num_matches)

        # remove completed champions for efficiency
        champion_names.remove(champion)

    return duo_wr_dict


def team_comp_wr(team_comp: tuple[str, str, str, str, str]) -> tuple[float, int] | None:
    """
    Return the winrate of a team comp and number of games played, None if no games with this comp.
    """
    # encode champions
    champs: list[uint8] = [lib.encoded_champ_id.name_to_int(champ) for champ in team_comp]

    # select all games where all were picked
    games = duckdb.sql(f"select * from _dataset where list_contains(picks, {champs[0]}) and list_contains(picks, {champs[1]}) and list_contains(picks, {champs[2]}) and list_contains(picks, {champs[3]}) and list_contains(picks, {champs[4]})")

    # only ones where all champions are on the same team
    for champ in champs[1:]:
        games = games.filter(f"list_contains(picks[1:5], {champs[0]}) = list_contains(picks[1:5], {champ})")

    total_games = games.count("win").fetchone()[0]

    # count wins
    wins = games.filter(f"(list_contains(picks[1:5], {champs[0]}) and win) or (list_contains(picks[6:10], {champs[0]}) and not win)")
    wins = wins.count("win").fetchone()[0]

    # calculate losses
    losses = total_games - wins

    if total_games > 0:
        return wins / (wins + losses), total_games
    else:
        return 0, total_games


if __name__ == "__main__":
    print(_dataset.count_rows())
    print(team_comp_wr(("Mordekaiser", "Viego", "Sylas", "Caitlyn", "Lux")))

    # champ = "Kayle"
    # alternate = "Volibear"
    # bs_wr, rs_wr, total_wr, num_matches = champ_winrate(champ)
    # print(f"{champ}: blue side winrate: {bs_wr:.2%}, red side winrate: {rs_wr:.2%}, total winrate: {total_wr:.2%}, num of matches: {num_matches}")
    # bs_wr, rs_wr, total_wr, num_matches = champ_winrate(alternate)
    # print(f"{alternate}: blue side winrate: {bs_wr:.2%}, red side winrate: {rs_wr:.2%}, total winrate: {total_wr:.2%}, num of matches: {num_matches}")
    # syn_wr, num_matches = synergy(champ, alternate)
    # print(f"{champ} + {alternate}: total winrate {syn_wr:.2%}, num of matches: {num_matches}")
    # print("\n\n")

    #print(champ_duo_winrate("Kayle"))
    print(all_duo_winrates())
    #print(team_comp_wr(("Mordekaiser", "Viego", "Sylas", "Caitlyn", "Lux")))

    # temp = all_duo_winrates()
    # save_file = open('all_champions_duo_wr_pickle.db', 'wb')
    # pickle.dump(temp, save_file)
    # save_file.close()

    # for champ in temp:
    #     print("Synergies with: ", champ)
    #     for alternate in temp[champ]:
    #         print(alternate)

