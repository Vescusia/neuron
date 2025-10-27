from time import time
from typing import Self

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

    def count(self) -> int:
        """
        :return: number of records in this database.
        """
        with self.begin() as txn:
            return txn.stat()['entries']

    def clone_to(self, other: Self) -> tuple[int, int]:
        """
        Clone all entries from ``self`` to ``other``.

        :return: tuple (total number of records sourced (effectively ``self.count()``), total number of records written to ``other`` (no overwrite!)
        """
        total_sourced = 0
        total_written = 0

        with self.begin() as source_txn:
            with other.begin(write=True) as dest_txn:
                with dest_txn.cursor() as dest_cursor:
                    with source_txn.cursor() as source_cursor:
                        for key, value in source_cursor:
                            total_sourced += 1
                            total_written += int(dest_cursor.put(key, value, overwrite=False))  # will return False if the key already existed.

        return total_sourced, total_written

    def __str__(self):
        return f"RegionDB({self.continent})"

    def __repr__(self):
        return str(self)


class SummonerDB:
    """
    ContinentDB interface specifically for summoners.
    """

    def __init__(self, continent_db: ContinentDB):
        self._cdb = continent_db
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
        :return: ``True`` if it was written, or ``False`` to indicate the Summoner was already present and ``overwrite=False``. On success, the cursor is positioned on the new record.
        """
        key, value = self._encode_entry(puuid, next_fetch_time, wait_time)

        with self._cdb.begin(write=True) as txn:
            return txn.put(key, value, overwrite)

    def get(self, puuid: str) -> tuple[int, float] | None:
        puuid = EncodedPUUID(puuid)
        with self._cdb.begin() as txn:
            return self._decode_entry(txn.get(puuid))

    def put_multi(self, entries: list[tuple[str, int, int]], overwrite=False) -> tuple[int, int]:
        """
        :param entries: list of tuples (puuid, next fetch time, wait time)
        :param overwrite: whether to overwrite existing entries or not
        :return: a tuple (consumed, added), where consumed is the length of summoners, and added is the number of new summoners actually added to the database. added may be less than consumed when overwrite=False.
        """
        encoded = [self._encode_entry(s, t, w) for s, t, w in entries]

        with self._cdb.begin(write=True) as txn:
            cur = txn.cursor()
            return cur.putmulti(encoded, overwrite=overwrite)

    def get_multi(self, puuids: list[str]) -> list[tuple[str, int, int]]:
        """
        :return: list of tuples (puuid, time, wait time) for each summoner that is present in the database.
        """
        puuids = [EncodedPUUID(puuid) for puuid in puuids]

        with self._cdb.begin() as txn:
            cur = txn.cursor()
            items = cur.getmulti(puuids)

        decoded = [(EncodedPUUID(k).decode(), *self._decode_entry(v)) for k, v in items if v is not None]
        return decoded

    def expired_summoner(self) -> tuple[str, int, int] | None:
        """
        find and return a summoner whose next fetch time is in the past.
        :return: tuple (puuid, next fetch time, wait time) or None if no summoner is found.
        """
        now = time()

        with self._cdb.begin() as txn:
            cur = txn.cursor()
            for key, value in cur:
                fetch_time, wait_time = self._decode_entry(value)
                if fetch_time < now:
                    return EncodedPUUID(key).decode(), fetch_time, wait_time

            return None

    def count(self) -> int:
        """
        Count the number of summoners in this database.
        """
        return self._cdb.count()

    def __str__(self):
        return f"SummonerDB({self.continent})"

    def __repr__(self):
        return str(self)


class MatchDB:
    """
    ContinentDB interface specifically for marking matches as explored.

    'Mark' simply means that this lmdb database effectively only stores keys without values.
    Only the presence of the key (match id) is relevant.
    """

    def __init__(self, continent_db: ContinentDB):
        self._cdb = continent_db
        self.continent = continent_db.continent

    def mark(self, match_id: str) -> bool:
        """
        Marks the match as explored.

        :return: True if the match was not marked yet, False if it was.
        """
        encoded_id = lib.encoded_match_id.to_bytes(match_id)

        with self._cdb.begin(write=True) as txn:
            return txn.put(encoded_id, None)

    def is_marked(self, match_id: str) -> bool:
        """
        Checks if the match is marked as explored.
        """
        encoded_id = lib.encoded_match_id.to_bytes(match_id)

        with self._cdb.begin() as txn:
            return txn.get(encoded_id) is not None

    def filter_marked(self, match_ids: list[str]) -> list[str]:
        """
        Filters out unmarked matches from the given list of match ids.
        :return: Sublist of match ids that are marked.
        """
        encoded_ids = [lib.encoded_match_id.to_bytes(match_id) for match_id in match_ids]

        with self._cdb.begin() as txn:
            cur = txn.cursor()
            items = cur.getmulti(encoded_ids)

        selected = [lib.encoded_match_id.to_match_id(k) for k, v in items]
        return selected

    def count(self) -> int:
        """
        Count the number of marked matches in this database.
        """
        return self._cdb.count()

    def __str__(self):
        return f"MatchDB({self.continent})"

    def __repr__(self):
        return str(self)
