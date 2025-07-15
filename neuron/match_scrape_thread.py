from pathlib import Path
import time
from queue import Queue
import lzma
import os

import cassiopeia as cass

from continent_db import ContinentDB
from match_id_coder import MatchIdCoder
from encoded_puuid import EncodedPUUID
from syn_db import CodedSynergy
from syn_db import SynergyDB


def scrape_continent(stop_q: Queue[None], state_q: Queue[int], match_db: ContinentDB, sum_db: ContinentDB, syn_db: SynergyDB, matchup_db: SynergyDB, matches_path: Path) -> None:
    match_coder = MatchIdCoder(match_db)
    inc_explored_matches = 0
    total_explored_matches = 0

    # create matches_dir for json files
    matches_path = matches_path.joinpath(str(match_db.continent.value)).joinpath(f"{int(time.time())}.xz")
    if not os.path.exists(matches_path.parent):
        os.makedirs(matches_path.parent)
    # open file
    with lzma.open(matches_path, "w", preset=6) as matches_fp:
        while True:
            # break if we get the signal to stop
            if not stop_q.empty():
                stop_q.get()
                total_explored_matches += inc_explored_matches
                break

            # search for unexplored match
            with match_db.begin() as txn:
                for match_id, visited in txn.cursor():
                    if not visited:
                        new_match_id = match_id
                        break
                else:
                    # request match history from an unexplored player
                    explore_player(match_db, sum_db, match_coder, stop_q)
                    # update explored matches
                    total_explored_matches += inc_explored_matches
                    state_q.put(inc_explored_matches)
                    inc_explored_matches = 0
                    continue

            # request Match from RiotAPI
            id_int, id_region = match_coder.decode(new_match_id)
            new_match: cass.Match = cass.get_match(id_int, id_region)

            # explore match
            # explore synergies
            with syn_db.begin(new_match.patch, write=True) as stxn:
                with matchup_db.begin(new_match.patch, write=True) as mtxn:
                    for i in range(0, len(new_match.participants)):
                        for j in range(i, len(new_match.participants)):
                            p0, p1 = new_match.participants[i], new_match.participants[j]
                            win = 1 if p0.team.win else 0

                            if p0.team != p1.team:
                                comp = CodedSynergy(p0.champion, p1.champion, win, 1, syn_db)
                                txn = stxn
                            else:
                                comp = CodedSynergy(p0.champion, p1.champion, win, 1, matchup_db)
                                txn = mtxn

                            # insert into database
                            key, _ = comp.to_bytes()
                            old = txn.get(key)
                            if old:
                                old = CodedSynergy.from_bytes(key, old, comp.db)
                                comp.total += old.total
                                comp.wins += old.wins
                            txn.put(*comp.to_bytes())

            # save match as json
            _ = new_match.participants, new_match.teams, new_match.is_remake  # we have to actually load the data (see cassiopeia ghost loading)
            # to json
            match_json = new_match.to_json()
            match_json = match_json.encode()
            # save to file
            matches_fp.write(len(match_json).to_bytes(4, "big", signed=False))
            matches_fp.write(match_json)

            # add participants to SummonerDB
            with sum_db.begin(write=True) as txn:
                for participant in new_match.participants:
                    puuid = bytes(EncodedPUUID(participant.summoner.puuid))
                    matches = int.from_bytes(txn.get(puuid) or bytes(0), "big")
                    txn.put(
                        puuid,
                        max(0, matches).to_bytes(1, "big")
                    )

            # mark Match as explored
            with match_db.begin(write=True) as txn:
                txn.put(new_match_id, bytes(True))
                inc_explored_matches += 1

    # rename file to amount of matches scraped
    matches_path_new = matches_path.parent.joinpath(matches_path.name.removesuffix(".xz") + f"_{total_explored_matches}.xz")
    matches_path.rename(matches_path_new)


def explore_player(match_db: ContinentDB, sum_db: ContinentDB, match_coder: MatchIdCoder, stop_q: Queue[None]) -> None:
    # search for player with no explored matches
    try:
        with sum_db.begin(write=True) as txn:
            cur = txn.cursor()
            for puuid, explored_matches in cur:
                if explored_matches == (0).to_bytes(1, "big"):
                    puuid = EncodedPUUID(puuid)
                    break
            else:
                # break out of with statement to collect new players from the League API
                print(f"\n\n{sum_db} Ran out of players, getting new ones!\n")
                raise LookupError

    except LookupError:
        # get new players from challenger leagues
        puuids = fetch_players_from_league(
            sum_db.continent)  # this will absolutely demolish our rate limit and take ages

        # insert into database
        encoded_puuids = [(bytes(EncodedPUUID(puuid)), (0).to_bytes(1)) for puuid in puuids]
        with sum_db.begin(write=True) as txn:
            cur = txn.cursor()
            _, added = cur.putmulti(encoded_puuids, overwrite=False)

        # check if we actually inserted unexplored PUUIDs
        if added == 0:
            # fuck
            stop_q.put(None)
            raise Exception(f"\n\n\n\n[ERROR] {sum_db} Completely RAN OUT OF SUMMONERS\n\n")
        else:
            print("\nAdded", added, "unexplored players!")
            return explore_player(match_db, sum_db, match_coder, stop_q)

    # get their match history
    matches: list[cass.Match] = cass.get_match_history(sum_db.continent, puuid.decode(), queue=cass.Queue.ranked_solo_fives, count=100)

    # insert into match database
    with match_db.begin(write=True) as txn:
        for match in matches:
            match_id = match_coder.encode(match.id, match.region)
            txn.put(match_id, bytes(False), overwrite=False)

    # update summoner
    with sum_db.begin(write=True) as txn:
        txn.put(bytes(puuid), (len(matches) + 1).to_bytes(1, "big"), overwrite=True)
    return None


# collect the players from the grandmaster leagues of all regions in our continent
def fetch_players_from_league(continent: cass.data.Continent) -> list[str]:
    # find regions in our continent
    puuids = []
    for region in cass.Region:
        if region.continent == continent:
            print("\nGetting Players of Challenger League for", region.name)
            gms = cass.get_challenger_league(cass.Queue.ranked_solo_fives, region).entries
            # extract the summoners puuid from the entries
            [puuids.append(e.summoner.puuid) for e in gms]
            # Legally Incorporated Distribution and Logistics company

    return puuids
