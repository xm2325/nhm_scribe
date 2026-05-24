from __future__ import annotations

import argparse
from herbarium_scribe.pipeline import stage_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo_10.yaml")
    args = parser.parse_args()
    result = stage_graph(args.config)
    print(result)


if __name__ == "__main__":
    main()
