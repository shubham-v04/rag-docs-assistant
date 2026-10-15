"""Ask one question from the command line and show how the graph handled it."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.builder import run_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        result = run_graph(args.question)
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    print(f"trace:        {' -> '.join(result['trace'])}")
    print(f"query type:   {result['query_type']}")
    if result["standalone_question"] != args.question:
        print(f"standalone:   {result['standalone_question']}")
    print(f"final query:  {result['search_query']}")
    print(f"retries:      {result['retry_count']}")
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("sources:")
        for src in result["sources"]:
            print(f"  - {src['source']} | {src['header_path']}")


if __name__ == "__main__":
    main()
