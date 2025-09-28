from pathlib import Path
from os import remove
from time import time

from numpy import uint8
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import lib


class WriteError:
    class MissingVersion(Exception):
        """
        Match does not contain a game version (['info']['gameVersion']).
        """
        def __init__(self):
            pass

    class InvalidChampionID(Exception):
        """
        The 'championId' of a participant does not match any champion.
        """
        def __init__(self, message):
            self.message = message


schema = pa.schema(
            [
                ("patch", pa.uint16()),
                ("ranked_score", pa.uint8()),
                ("win", pa.bool_()),
                ("picks", pa.list_(pa.uint8(), 10)),
                ("bans", pa.list_(pa.uint8(), 10)),
            ]
        )


class ContinentDatasetWriter:
    """
    A continent-specific Dataset Writer. Please do not construct this class directly; use LolDatasetWriter instead.
    """

    def __init__(self, _base_path: Path, _continent: str, _write_interval: int):
        self.continent = _continent
        self.write_interval = _write_interval

        # add the continent to the base path
        self.base_path = _base_path / _continent

        # create the match list for the write_interval
        self.match_list = []

    def write_match(self, match: dict, ranked_score: uint8) -> None | WriteError:
        """
        Add a match to the cached table.
        Once the table has reached a certain size (write_interval in LolDataset), it will be written to disk.
        """
        # encode patch
        patch = match['info']['gameVersion']
        if not patch:
            raise WriteError.MissingVersion
        encoded_patch = lib.encoded_patch.to_int(patch)

        # extract if the blue side won
        win: bool = match['info']['teams'][0]['win']

        # extract the bans
        bans = []
        bans.extend(match['info']['teams'][0]['bans'])
        bans.extend(match['info']['teams'][1]['bans'])
        # sort them by pick order
        bans.sort(key=lambda ban: ban['pickTurn'])

        # encode bans to u8
        try:
            encoded_bans = [lib.encoded_champ_id.to_int(ban['championId']) for ban in bans]
        except KeyError as e:
            raise WriteError.InvalidChampionID(str(e))

        # extract the picked champions
        # there seems to be no indication for pick order other than... participant order?
        picks = match['info']['participants']
        picks = [participant['championId'] for participant in picks]

        # encode picks to u8
        try:
            encoded_picks = [lib.encoded_champ_id.to_int(pick) for pick in picks]
        except KeyError as e:
            raise WriteError.InvalidChampionID(str(e))

        # map to schema
        row = [encoded_patch, ranked_score, win] + [encoded_picks] + [encoded_bans]
        row = {schema.field(i).name: col for i, col in enumerate(row)}

        # append row to the match list
        self.match_list.append(row)

        # write to disk if the interval is reached
        if len(self.match_list) >= self.write_interval:
            self.write_match_list()

    def write_match_list(self):
        """
        Write the match list to disk.
        This method is called automatically when the write_interval is reached but should be called manually otherwise.
        :return:
        """
        table = pa.Table.from_pylist(self.match_list, schema=schema)

        pq.write_to_dataset(table, self.base_path, basename_template=f"{len(self.match_list)}_{int(time())}_0_" + "{i}.pq")
        self.match_list.clear()

    def close(self, redistribute: int | None = None):
        """
        Close the writer, writing the remaining matches to disk.
        :param redistribute: If not None, will redistribute the parquet files so that they hold this number of matches (reduces number of files on disk).
        """
        # write remaining matches to disk
        self.write_match_list()

        # redistribute the parquet files if wanted
        if redistribute is not None:
            self._redistribute_parquet(redistribute)

    def _redistribute_parquet(self, num_matches: int):
        """
        Redistribute the parquet files to a larger number of matches than write_interval.
        This is useful when there are many small parquet files.
        :param num_matches: Number of matches the parquet files should hold.
        """
        # unify time in the name of the written parquet files
        current_time = int(time())

        # keep track of the number of parquet files we wrote
        num_pq_written = 0

        # create a dictionary that will accumulate matches
        matches = {}

        # open the dataset, ignoring parquet files that already contain num_matches
        dataset = open_dataset(self.base_path, ignore_prefixes=[str(num_matches)])

        # get batches (every batch is maximally only as big as the file)
        for batch in dataset.to_batches(batch_size=num_matches):
            batch: pa.RecordBatch = batch

            # insert batch into matches
            for name, column in batch.to_pydict().items():
                try:
                    matches[name].extend(column)
                except KeyError:
                    matches[name] = column

            # write accumulated matches to larger parquets
            while len(matches['win']) >= num_matches:
                # build pyarrow table from matches
                table: pa.Table = pa.Table.from_pydict(matches, schema=schema).slice(length=num_matches)

                # write accumulated matches to a larger parquet file (only up to num_matches)
                # suffix the accumulated datasets with '.tmp', prefix with num_matches
                pq.write_to_dataset(table, self.base_path,
                                    basename_template=f"{table.num_rows}_{current_time}_{num_pq_written}" + "_{i}.pq.tmp"
                                    )
                num_pq_written += 1

                # remove written matches
                for name in matches.keys():
                    matches[name] = matches[name][num_matches:]

        # write residual matches
        table: pa.Table = pa.Table.from_pydict(matches, schema=schema)
        # suffix the accumulated datasets with '.tmp'
        pq.write_to_dataset(table, self.base_path,
                            basename_template=f"{table.num_rows}_{current_time}_{num_pq_written}" + "_{i}.pq.tmp"
                            )

        # delete the parquet files that we just redistributed (without the suffix '.tmp' or the prefix str(num_matches))
        for file in self.base_path.glob('*.pq'):
            if file.name.startswith(str(num_matches)):
                continue
            # DANGEROUS!
            remove(file)

        # remove the suffix '.tmp' from the redistributed files
        for file in self.base_path.glob('*.tmp'):
            file.rename(file.with_name(file.name.removesuffix('.tmp')))


class LolDatasetWriter:
    """
    Dataset writing class. This class is meant to be used as a context manager.
    Use .open_continent() to open a continent-specific DatasetWriter.

    Uses the parquet format.
    :param base_path: The base path for the dataset. It will be created if it doesn't exist.
    :param write_interval: The number of matches that will be cached before being written to disk.
    """

    def __init__(self, base_path: Path, write_interval: int = 1_000):
        # define schema for the dataset
        self.base_path = base_path
        self.write_interval = write_interval

    def open_continent(self, continent: str) -> ContinentDatasetWriter:
        return ContinentDatasetWriter(self.base_path, continent, self.write_interval)


def open_dataset(base_path: str | Path, ignore_prefixes: list[str] = None) -> ds.Dataset:
    return ds.dataset(base_path, format="parquet", partitioning="hive", schema=lib.league_of_parquet.schema, ignore_prefixes=ignore_prefixes)
