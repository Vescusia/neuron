from pathlib import Path

from numpy import uint8
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import lib


class ContinentDataset:
    """
    A continent-specific Dataset. Please do not construct this class directly; use LolDataset instead.
    """

    def __init__(self, _schema: pa.Schema, _base_path: Path, _continent: str, _write_interval: int):
        self.schema = _schema
        self.continent = _continent
        self.write_interval = _write_interval

        # add the continent to the base path
        self.base_path = _base_path / _continent

        # create the match list for the write_interval
        self.match_list = []

    def append(self, match: dict, ranked_score: uint8):
        """
        Append a match to the table.
        Once the table has reached a certain size (write_interval in LolDataset), it will be written to disk.
        """
        # encode patch
        encoded_patch = lib.encoded_patch.to_int(match['info']['gameVersion'])

        # extract if the blue side won
        win: bool = match['info']['teams'][0]['win']

        # extract the bans
        bans = []
        bans.extend(match['info']['teams'][0]['bans'])
        bans.extend(match['info']['teams'][1]['bans'])
        # sort them by pick order
        bans.sort(key=lambda ban: ban['pickTurn'])
        # map to encoded champions
        encoded_bans = [lib.encoded_champ_id.to_int(ban['championId']) for ban in bans]

        # extract the picked champions
        # there seems to be no indication for pick order other than... participant order?
        picks = match['info']['participants']
        picks = [participant['championId'] for participant in picks]
        encoded_picks = [lib.encoded_champ_id.to_int(pick) for pick in picks]

        # map to schema
        row = [encoded_patch, ranked_score, win] + encoded_picks + encoded_bans
        row = {self.schema.field(i).name: col for i, col in enumerate(row)}

        # append row to the match list
        self.match_list.append(row)

        # write to disk if the interval is reached
        if len(self.match_list) >= self.write_interval:
            self._write_match_list()

    def _write_match_list(self):
        table = pa.Table.from_pylist(self.match_list, schema=self.schema)
        self.match_list.clear()

        pq.write_to_dataset(table, self.base_path, partition_cols=["patch"])


class LolDataset:
    """
    Overarching dataset class. For writing, use open_continent/ContinentDataset instead.
    :param base_path: The base path for the dataset. It will be created if it doesn't exist.
    :param write_interval: The number of matches that will be cached before being written to disk.
    """

    def __init__(self, base_path: Path, write_interval: int = 1_000):
        # define schema for the dataset
        self.schema = pa.schema(
            [
                ("patch", pa.uint16()),
                ("ranked_score", pa.uint8()),
                ("win", pa.bool_()),
            ]
            + [(f"pick{i}", pa.uint8()) for i in range(10)]
            + [(f"ban{i}", pa.uint8()) for i in range(10)]
        )
        self.base_path = base_path
        self.write_interval = write_interval

    def open_continent(self, continent: str) -> ContinentDataset:
        return ContinentDataset(self.schema, self.base_path, continent, self.write_interval)
