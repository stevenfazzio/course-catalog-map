"""Stage 00: fetch one university's raw catalog into data/<source>/raw/.

Resumable: a department whose file already exists is not re-downloaded. Delete a file to refetch it.
"""

import argparse

import config
from sources import get_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    args = ap.parse_args()
    get_source(args.source).fetch_raw(config.source_paths(args.source))


if __name__ == "__main__":
    main()
