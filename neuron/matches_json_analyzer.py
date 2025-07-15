import json
import lzma
import sys
from pathlib import Path


def main() -> None:
    path = None if len(sys.argv) == 1 else sys.argv[1]
    if not path:
        print("PATH Argument needed | Path to compressed matches file")
        print("Usage: matches_json_analyzer.py PATH")
        return

    path = Path(path)
    if not path.exists():
        print(f"Path '{path}' does not exist!")
        return

    print(f"Opening '{path}'")
    with lzma.open(path, "rb") as f:
        while True:
            # read size of match
            size = f.read(4)
            if size == b'':
                break
            size = int.from_bytes(size, "big", signed=False)
            print(size)

            # read match
            match = json.loads(f.read(size))


if __name__ == "__main__":
    main()
