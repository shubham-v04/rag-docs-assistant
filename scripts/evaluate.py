"""Run the evaluation questions through the graph and report how it did.

Needs GROQ_API_KEY and an ingested index. Results go to eval_results.json, which is
rewritten after every question so an interrupted run keeps what it has. It waits between
questions (--pause, or EVAL_PAUSE; default 5 seconds) to stay under per-minute rate limits.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.graph.builder import run_graph

QUESTIONS_PATH = ROOT / "data" / "eval_questions.json"
RESULTS_PATH = ROOT / "eval_results.json"
EXPECTATIONS = {"answer", "no_answer"}
QUESTION_WIDTH = 52


def load_questions(path: Path) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, item in enumerate(items, start=1):
        if not item.get("question"):
            raise ValueError(f"question {i} has no text")
        if item.get("expect") not in EXPECTATIONS:
            raise ValueError(f"question {i}: expect must be 'answer' or 'no_answer'")
        if item["expect"] == "answer" and not item.get("expected_source"):
            raise ValueError(
                f"question {i}: answerable questions need an expected_source"
            )
    return items


def score_result(
    item: dict, output: dict | None, latency: float, error: str | None = None
) -> dict:
    """Turn one graph run (or its error) into a flat result record."""
    if error or output is None:
        actual, sources = "error", []
    else:
        actual = "no_answer" if output["no_answer"] else "answer"
        sources = list(dict.fromkeys(s["source"] for s in output["sources"]))

    expected_source = item.get("expected_source")
    source_found = None
    if item["expect"] == "answer":
        source_found = expected_source in sources

    output = output or {}
    return {
        "question": item["question"],
        "expect": item["expect"],
        "actual": actual,
        "correct": actual == item["expect"],
        "retries": output.get("retry_count"),
        "sources": sources,
        "expected_source": expected_source,
        "source_found": source_found,
        "latency_seconds": round(latency, 2),
        "query_type": output.get("query_type"),
        "grounded": output.get("grounded"),
        "trace": output.get("trace"),
        "answer": output.get("answer"),
        "error": error,
    }


def run_question(item: dict, run: Callable[..., dict]) -> dict:
    start = time.perf_counter()
    try:
        output = run(item["question"])
    except Exception as exc:
        # one bad question should not lose the rest of the run
        return score_result(
            item, None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
        )
    return score_result(item, output, time.perf_counter() - start)


def summarize(results: list[dict]) -> dict:
    answerable = [r for r in results if r["expect"] == "answer"]
    latencies = [r["latency_seconds"] for r in results]
    return {
        "total": len(results),
        "routing_correct": sum(r["correct"] for r in results),
        "answerable": len(answerable),
        "source_found": sum(bool(r["source_found"]) for r in answerable),
        "errors": sum(r["actual"] == "error" for r in results),
        "average_latency_seconds": round(sum(latencies) / len(latencies), 2)
        if latencies
        else 0.0,
    }


def summary_line(summary: dict) -> str:
    line = (
        f"correct routing {summary['routing_correct']}/{summary['total']}, "
        f"expected source found {summary['source_found']}/{summary['answerable']}"
    )
    if summary["errors"]:
        line += f", {summary['errors']} failed with an error"
    return line


def format_table(results: list[dict]) -> str:
    headers = [
        "#",
        "question",
        "expected",
        "actual",
        "retries",
        "sources",
        "source found",
        "latency",
    ]
    rows = []
    for i, r in enumerate(results, start=1):
        question = r["question"]
        if len(question) > QUESTION_WIDTH:
            question = question[: QUESTION_WIDTH - 1] + "…"
        found = {True: "yes", False: "no", None: "-"}[r["source_found"]]
        rows.append(
            [
                str(i),
                question,
                r["expect"],
                r["actual"],
                "-" if r["retries"] is None else str(r["retries"]),
                ", ".join(r["sources"]) or "-",
                found,
                f"{r['latency_seconds']:.1f}s",
            ]
        )

    widths = [
        max(len(row[col]) for row in [headers, *rows]) for col in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) for cell, width in zip(cells, widths)
        ).rstrip()

    return "\n".join([line(headers), line(["-" * w for w in widths]), *map(line, rows)])


def require_ready() -> None:
    if not get_settings().groq_api_key:
        sys.exit(
            "GROQ_API_KEY is not set. Export it and run again: export GROQ_API_KEY=..."
        )
    # imported here so the key check above does not wait for the embedding stack
    from app.ingestion.indexer import chunk_count

    if chunk_count() == 0:
        sys.exit("The index is empty. Run python scripts/ingest.py first.")


def save_results(path: Path, results: list[dict], run_at: str, complete: bool) -> None:
    payload = {
        "run_at": run_at,
        "complete": complete,
        "summary": summarize(results),
        "results": results,
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # replace in one step, so stopping the script mid-write cannot corrupt the file
    os.replace(tmp, path)


def progress_line(position: str, result: dict) -> str:
    if result["error"]:
        return f"{position} ERROR after {result['latency_seconds']:.1f}s: {result['error']}"
    found = {True: "yes", False: "no", None: "-"}[result["source_found"]]
    return (
        f"{position} {result['actual']} (expected {result['expect']}), "
        f"retries {result['retries']}, source found {found}, {result['latency_seconds']:.1f}s"
    )


def main(
    argv: list[str] | None = None,
    run: Callable[..., dict] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--pause",
        type=float,
        default=None,
        help="seconds to wait between questions (default: EVAL_PAUSE, else 5)",
    )
    args = parser.parse_args(argv)
    pause = max(0.0, get_settings().eval_pause if args.pause is None else args.pause)

    if run is None:
        require_ready()
        run = run_graph

    questions = load_questions(args.questions)
    run_at = datetime.now(UTC).isoformat(timespec="seconds")
    results: list[dict] = []
    try:
        for i, item in enumerate(questions, start=1):
            position = f"[{i}/{len(questions)}]"
            if i > 1 and pause:
                print(f"waiting {pause:g}s before the next question", file=sys.stderr)
                sleep(pause)
            print(f"{position} running: {item['question']}", file=sys.stderr)
            results.append(run_question(item, run))
            save_results(args.output, results, run_at, complete=False)
            print(progress_line(position, results[-1]), file=sys.stderr)
    except KeyboardInterrupt:
        print(
            f"\ninterrupted after {len(results)} of {len(questions)} questions",
            file=sys.stderr,
        )
    save_results(args.output, results, run_at, complete=len(results) == len(questions))

    print(f"\n{format_table(results)}\n")
    for r in results:
        if r["error"]:
            print(f"error: {r['question']!r}: {r['error']}")
    print(summary_line(summarize(results)))
    print(f"results saved to {args.output}")


if __name__ == "__main__":
    main()