from pathlib import Path
from os.path import exists
from os import rename
from json import dumps
import gzip  # lz4 at 6 -> ~45 MB @ 10_000 matches


class CompressedJSONBall:
    """
    A zlib (gzip) compressed JSON 'Ball', meaning a single file with multiple JSON objects.
    """

    def __init__(self, path: Path, append_to_file_name=True, split_every: int | None = 100_000, _splits: int = 0):
        """
        :param path: path to the file. it must not exist yet.
        :param append_to_file_name: append the number of objects within the ball to the file name.
        :param split_every: split the ball into another file every N objects.
        """
        assert not exists(path)

        self.path = path
        self._original_path = path
        self._num_of_objects = 0
        self._file = gzip.open(path, "w", compresslevel=7)
        self.append_to_file_name = append_to_file_name
        self.split_every = split_every
        self._num_of_splits = 0
        self._splits = _splits

    def append(self, obj):
        """
        append an object to the ball.
        """
        # encode JSON object
        obj = dumps(obj).encode("utf-8")
        # write the length
        self._file.write(len(obj).to_bytes(4, "big", signed=False))
        # write object
        self._file.write(obj)
        self._num_of_objects += 1

        # finish writing to this file and open a new one
        if self.split_every is not None and self._num_of_objects >= self.split_every:
            # save the original path for the new file
            og_path = self._original_path

            # close the current file
            self.close()

            # rename the current file to indicate split
            self._original_path = self._original_path.parent.joinpath(f"split{self._splits}_{self._original_path.name}")
            self.append_number_of_objects(self._num_of_objects)

            # reinitialize to create a "new" ball
            self.__init__(og_path, append_to_file_name=self.append_to_file_name, split_every=self.split_every, _splits=self._splits+1)

    def close(self):
        self._file.close()

        # append the number of objects stored to the file name
        if self.append_to_file_name:
            self.append_number_of_objects(self._num_of_objects)

    def append_number_of_objects(self, num: int):
        # create the new path
        split_path_name = self._original_path.name.split('.')
        new_file_name = f"{split_path_name[0]}_{num}.{split_path_name[1]}"
        new_path = self.path.parent / new_file_name

        # rename the file
        rename(self.path, new_path)
        self.path = new_path
