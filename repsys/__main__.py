import logging
import os
import sys

import coloredlogs

from repsys.cli import repsys_group


def setup_logging(level):
    coloredlogs.install(
        level=level,
        use_chroot=False,
        fmt="%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
    )


def main():
    sys.path.insert(1, os.getcwd())
    setup_logging(logging.INFO)

    repsys_group(prog_name="repsys")


if __name__ == "__main__":
    main()
