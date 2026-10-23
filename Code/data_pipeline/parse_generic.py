"""Parse raw repository records into requirement, test, and source units."""

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.common_parse import process_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reader = gzip.open if args.input.suffix == ".gz" else open
    writer = gzip.open if args.output.suffix == ".gz" else open
    with reader(args.input, "rt", encoding="utf-8") as source, \
            writer(args.output, "wt", encoding="utf-8") as output:
        for line in source:
            if line.strip():
                record = process_instance(json.loads(line))
                output.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
