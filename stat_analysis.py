import duckdb

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
    total_games = games.count("win").fetchone()[0]

    # count wins
    wins = games.filter(f"(list_contains(picks[1:5], {champ0_int}) and win) or (list_contains(picks[6:10], {champ0_int}) and not win)")
    wins = wins.count("win").fetchone()[0]

    # calculate losses
    losses = total_games - wins

    return wins / (wins + losses), total_games


if __name__ == "__main__":
    print(_dataset.count_rows())

    champ = "Ahri"
    alternate = "Caitlyn"

    bs_wr, rs_wr, total_wr, num_matches = champ_winrate(champ)
    print(f"{champ}: blue side winrate: {bs_wr:.2%}, red side winrate: {rs_wr:.2%}, total winrate: {total_wr:.2%}, num of matches: {num_matches}")
    bs_wr, rs_wr, total_wr, num_matches = champ_winrate(alternate)
    print(f"{alternate}: blue side winrate: {bs_wr:.2%}, red side winrate: {rs_wr:.2%}, total winrate: {total_wr:.2%}, num of matches: {num_matches}")
    syn_wr, num_matches = synergy(champ, alternate)
    print(f"{champ} + {alternate}: total winrate {syn_wr:.2%}, num of matches: {num_matches}")
