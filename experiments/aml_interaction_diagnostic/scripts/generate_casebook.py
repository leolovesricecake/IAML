import argparse
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.casebook import build_casebook


def build_parser():
    parser = argparse.ArgumentParser(description="Generate a Markdown casebook for a diagnostic run")
    parser.add_argument("--run-dir", required=True)
    return parser


if __name__ == "__main__":
    build_casebook(build_parser().parse_args().run_dir)
