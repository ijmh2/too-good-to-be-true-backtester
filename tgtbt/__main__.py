"""Enables `python -m tgtbt <args>` as an alternative to the `tgtbt` console script."""

import sys

from tgtbt.cli import main

if __name__ == "__main__":
    sys.exit(main())
