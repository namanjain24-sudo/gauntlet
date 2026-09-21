"""Command line: ``python -m gauntlet run`` / ``python -m gauntlet report``."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import experiment, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gauntlet", description="Adversarial arena for prompt-injection defenses.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the experiment and write results.json + report.html")
    run.add_argument("--seeds", type=int, default=3)
    run.add_argument("--rounds", type=int, default=5)
    run.add_argument("--generations", type=int, default=12)
    run.add_argument("--pop", type=int, default=200)
    run.add_argument("--quick", action="store_true", help="1 seed, 3 rounds, small population (~1 min)")
    run.add_argument("--out", type=Path, default=Path("reports"))
    rep = sub.add_parser("report", help="re-render report.html from an existing results.json")
    rep.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)

    if args.cmd == "run":
        if args.quick:
            args.seeds, args.rounds, args.generations, args.pop = 1, 3, 8, 120
        res = experiment.run(args.seeds, args.rounds, args.generations, args.pop)
        experiment.save(res, args.out / "results.json")
    report.write(args.out / "results.json", args.out / "report.html")
    print(f"wrote {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
