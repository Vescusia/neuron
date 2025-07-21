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

    # filter out the blue side and red side picks respectively
    bs_games = games.filter(f"list_contains(picks[0:5], {champ_int})")
    rs_games = games.filter(f"list_contains(picks[5:10], {champ_int})")

    # count total games
    total_games = games.count("win").fetchone()[0]

    # count blue side wins/losses
    bs_wins = bs_games.filter("win").count("win").fetchone()[0]
    bs_losses = bs_games.filter("not win").count("win").fetchone()[0]

    # count red side wins/losses
    rs_wins = rs_games.filter("win").count("win").fetchone()[0]
    rs_losses = rs_games.filter("not win").count("win").fetchone()[0]

    return bs_wins / (bs_wins + bs_losses), rs_wins / (rs_wins + rs_losses), (bs_wins + rs_wins) / total_games, total_games


if __name__ == "__main__":
    champ = "LeeSin"

    bs_wr, rs_wr, total_wr, num_matches = champ_winrate(champ)

    print(f"{champ}: blue side winrate: {bs_wr:.2%}, red side winrate: {rs_wr:.2%}, total winrate: {total_wr:.2%}, num of matches: {num_matches}")
