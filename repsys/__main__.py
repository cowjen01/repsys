import os
import sys

from repsys.cli import repsys_group


def main():
    sys.path.insert(1, os.getcwd())
    repsys_group(prog_name="repsys")


if __name__ == "__main__":
    main()
