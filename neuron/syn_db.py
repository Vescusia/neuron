from typing import Iterable, Self

import cassiopeia
import lmdb


class SynergyDB:
    def __init__(self, path: str, max_dbs=50, **kwargs):
        self.env = lmdb.open(path, max_dbs=max_dbs, **kwargs)

        # create champion map
        self._champ_map = {champ: i for i, champ in enumerate(cassiopeia.get_champions("EUW"))}

        # insert into or update from database
        new_inserted = 0
        with self.env.begin(write=True) as txn:
            for champ, i in self._champ_map.items():
                prev = txn.get(champ.id.to_bytes(4, 'big', signed=False))
                if prev:
                    self._champ_map[champ] = int.from_bytes(prev, byteorder='big', signed=False)
                else:
                    new_inserted += 1
                    txn.put(champ.id.to_bytes(4, 'big', signed=False), i.to_bytes(1, byteorder='big', signed=False))

        self._rev_champ_map = {v: k for k, v in self._champ_map.items()}
        self._champ_map = {k.id: v for k, v in self._champ_map.items()}

    def begin(self, patch: cassiopeia.Patch, write: bool = False, **kwargs):
        db = self.env.open_db(patch.name.encode())
        return self.env.begin(write=write, db=db, **kwargs)

    @property
    def champ_map(self):
        return self._champ_map

    @property
    def rev_champ_map(self):
        return self._rev_champ_map


class CodedSynergy:
    def __init__(self, champ0: cassiopeia.Champion, champ1: cassiopeia.Champion, wins: int, total: int, champ_encoding: SynergyDB):
        self.wins = wins
        self.total = total

        self._champs = champ_encoding

        self._left = champ0
        self._right = champ1
        self._sort()

    @staticmethod
    def from_bytes(coded_key: bytes, coded_value: bytes, champ_encoding: SynergyDB):
        # decode champions
        left = champ_encoding.rev_champ_map[coded_key[0]]
        right = champ_encoding.rev_champ_map[coded_key[1]]
        # decode wins/total
        wins = int.from_bytes(coded_value[0:2], byteorder='big', signed=False)
        total = int.from_bytes(coded_value[2:4], byteorder='big', signed=False)

        return CodedSynergy(left, right, wins, total, champ_encoding)

    @staticmethod
    def all(champ0: cassiopeia.Champion, champ_encoding: SynergyDB) -> Iterable:
        for champ1 in champ_encoding.rev_champ_map.values():
            if champ0 != champ1:
                yield CodedSynergy(champ0, champ1, 0, 0, champ_encoding)

    # to key and value bytes
    def to_bytes(self) -> tuple[bytes, bytes]:
        key = bytearray()
        # code champions
        key.extend(self.db.champ_map[self.left.id].to_bytes(1, byteorder='big', signed=False))
        key.extend(self.db.champ_map[self.right.id].to_bytes(1, byteorder='big', signed=False))
        # code wins
        value = bytearray()
        value.extend((self.wins or 0).to_bytes(2, byteorder='big', signed=False))
        # code total
        value.extend((self.total or 0).to_bytes(2, byteorder='big', signed=False))

        return key, value

    @property
    def left(self) -> cassiopeia.Champion:
        return self._left

    @property
    def right(self) -> cassiopeia.Champion:
        return self._right

    @property
    def winrate(self) -> float:
        return self.wins / max(self.total, 1)

    @property
    def db(self) -> SynergyDB:
        return self._champs

    def swapped(self) -> Self:
        new = self
        new._swap()
        return new

    def _swap(self):
        self._left, self._right = self._right, self._left
        self.wins = self.total - self.wins

    def _sort(self):
        # switch the left and right champion such that the left one has a smaller id
        if self.db.champ_map[self.left.id] > self.db.champ_map[self.right.id]:
            self._swap()

    def __str__(self) -> str:
        return f"[{self.left.name:14} <> {self.right.name:14} | {self.total:3}] -> {(self.winrate or 0)*100:.2f}% WR"
