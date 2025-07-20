from time import time

import numpy as np
from lmdb import Environment

import lib
from .encoded_puuid import EncodedPUUID


class ContinentDB:
    """
    A continent-specific LMDB Database within a lmdb.Environment.
    """

    def __init__(self, env: Environment, continent: str, **kwargs):
        self.continent = continent
        self.env = env
        self.db = env.open_db(str(continent).encode(), **kwargs)

    def begin(self, write: bool = False, **kwargs):
        return self.env.begin(write=write, db=self.db, **kwargs)

    def __str__(self):
        return f"RegionDB({self.continent})"

    def __repr__(self):
        return str(self)


class SummonerDB:
    """
    ContinentDB interface specifically for summoners.
    """

    def __init__(self, continent_db: ContinentDB):
        self.cdb = continent_db
        self.continent = continent_db.continent

    @staticmethod
    def _encode_entry(puuid: str, next_fetch_time: int, wait_time: int) -> tuple[bytes, bytes]:
        value = bytearray()
        value.extend(next_fetch_time.to_bytes(8, "big"))
        value.extend(wait_time.to_bytes(4, "big"))

        return EncodedPUUID(puuid).to_bytes(), value

    @staticmethod
    def _decode_entry(value: bytes | None) -> tuple[int, int] | None:
        if bytes is None:
            return None

        next_fetch_time = int.from_bytes(value[:8], "big")
        wait_time = int.from_bytes(value[8:], "big")

        return next_fetch_time, wait_time

    def put(self, puuid: str, next_fetch_time: int, wait_time: int, overwrite=False) -> bool:
        """
        :return: True if it was written, or False to indicate the Summoner was already present and overwrite=False. On success, the cursor is positioned on the new record.
        """
        key, value = self._encode_entry(puuid, next_fetch_time, wait_time)

        with self.cdb.begin(write=True) as txn:
            return txn.put(key, value, overwrite)

    def get(self, puuid: str) -> tuple[int, float] | None:
        puuid = EncodedPUUID(puuid)
        with self.cdb.begin() as txn:
            return self._decode_entry(txn.get(puuid))

    def put_multi(self, entries: list[tuple[str, int, int]], overwrite=False) -> tuple[int, int]:
        """
        :param entries: list of tuples (puuid, next fetch time, wait time)
        :param overwrite: whether to overwrite existing entries or not
        :return: a tuple (consumed, added), where consumed is the length of summoners, and added is the number of new summoners actually added to the database. added may be less than consumed when overwrite=False.
        """
        encoded = [self._encode_entry(s, t, w) for s, t, w in entries]

        with self.cdb.begin(write=True) as txn:
            cur = txn.cursor()
            return cur.putmulti(encoded, overwrite=overwrite)

    def get_multi(self, puuids: list[str]) -> list[tuple[str, int, int]]:
        """
        :return: list of tuples (puuid, time, wait time) for each summoner that is present in the database.
        """
        puuids = [EncodedPUUID(puuid) for puuid in puuids]

        with self.cdb.begin() as txn:
            cur = txn.cursor()
            keys_and_values = cur.getmulti(puuids)

        decoded = [(EncodedPUUID(k).decode(), *self._decode_entry(v)) for k, v in keys_and_values if v is not None]
        return decoded

    def expired_summoner(self) -> tuple[str, int, int] | None:
        """
        find and return a summoner whose next fetch time is in the past.
        :return: tuple (puuid, next fetch time, wait time) or None if no summoner is found.
        """
        now = time()

        with self.cdb.begin() as txn:
            cur = txn.cursor()
            for key, value in cur:
                fetch_time, wait_time = self._decode_entry(value)
                if fetch_time < now:
                    return EncodedPUUID(key).decode(), fetch_time, wait_time

            return None

    def __str__(self):
        return f"SummonerDB({self.continent})"

    def __repr__(self):
        return str(self)


class MatchDB:
    """
    ContinentDB interface specifically for matches.
    """

    def __init__(self, continent_db: ContinentDB):
        self.cdb = continent_db
        self.continent = continent_db.continent

    def put(self, match_id: str, explored: bool, tier: str, division: str, overwrite=False) -> bool:
        """
        :return: True if it was written, or False to indicate the Match was already present and overwrite=False. On success, the cursor is positioned on the new record.
        """
        key, value = self._encode_entry(match_id, explored, tier, division)

        with self.cdb.begin(write=True) as txn:
            return txn.put(key, value, overwrite=overwrite)

    def set_explored(self, match_id: str) -> bool:
        encoded_match_id = lib.encoded_match_id.to_bytes(match_id)

        with self.cdb.begin(write=True) as txn:
            value = txn.get(encoded_match_id)
            return txn.put(encoded_match_id, int.to_bytes(1), value[1], overwrite=True)

    @staticmethod
    def _encode_entry(match_id: str, explored: bool, tier: str, division: str) -> tuple[bytes, bytes]:
        encoded_match_id = lib.encoded_match_id.to_bytes(match_id)
        rank_int = lib.encoded_rank.to_int(tier, division)

        value = bytearray((int(explored), rank_int))

        return encoded_match_id, value

    @staticmethod
    def _decode_entry(value: bytes | None) -> tuple[bool, np.uint8] | None:
        if bytes is None:
            return None

        explored = bool(value[0])
        rank_int = np.uint8(np.frombuffer(value[1:], ">u1"))

        return explored, rank_int

    def get(self, match_id: str) -> tuple[bool, str, str] | None:
        encoded_match_id = lib.encoded_match_id.to_bytes(match_id)

        with self.cdb.begin() as txn:
            return self._decode_entry(txn.get(encoded_match_id))

    def put_multi(self, matches: list[tuple[str, bool, str, str]], overwrite=False) -> tuple[int, int]:
        """
        :param matches: list of tuples (match_id, explored, tier, division)
        :param overwrite: whether to overwrite existing entries or not
        :return: a tuple (consumed, added), where consumed is the length of match_ids, and added is the number of new matches actually added to the database. added may be less than consumed when overwrite=False.
        """
        keys_and_values = [self._encode_entry(match_id, fetched, tier, div) for match_id, fetched, tier, div in matches]

        with self.cdb.begin(write=True) as txn:
            cur = txn.cursor()
            return cur.putmulti(keys_and_values, overwrite=overwrite)

    def unexplored_match(self) -> tuple[str, np.uint8] | None:
        """
        find and return a match that has not been explored yet.

        :return: tuple (match_id, encoded rank) or None if no match is found.
        """
        with self.cdb.begin() as txn:
            cur = txn.cursor()
            for key, value in cur:
                explored, rank = self._decode_entry(value)
                if not explored:
                    return lib.encoded_match_id.to_match_id(key), rank

            return None

    def __str__(self):
        return f"MatchDB({self.continent})"

    def __repr__(self):
        return str(self)
