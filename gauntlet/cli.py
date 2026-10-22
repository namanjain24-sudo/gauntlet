"""Command line: ``python -m gauntlet run`` / ``python -m gauntlet report``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import attack, bench, cache, experiment, multilingual, report, scan


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
    run.add_argument("--kinds", default=",".join(experiment.KINDS),
                     help=f"comma-separated defenders from: {', '.join(experiment.ALL_KINDS)}")
    run.add_argument("--merge", action="store_true",
                     help="fold these defenders into the existing results.json instead of overwriting it")
    sc = sub.add_parser("scan", help="run your own prompts through the defenders (trains them on the fly, ~15s)")
    sc.add_argument("prompts", nargs="*", help="prompts to scan (or use --file / stdin)")
    sc.add_argument("--file", type=Path, help="text file with one prompt per line")
    sc.add_argument("--kinds", default=",".join(experiment.KINDS), help="defenders to use")
    sc.add_argument("--json", action="store_true", help="machine-readable output")
    at = sub.add_parser("attack", help="evolve attacks for YOUR goal against one defender and show what gets through")
    at.add_argument("goal", help="what the injection should make the model do, e.g. 'reveal your system prompt'")
    at.add_argument("--defender", default="hardened", help=f"one of: {', '.join(experiment.ALL_KINDS)}")
    at.add_argument("--generations", type=int, default=10)
    at.add_argument("--pop", type=int, default=120)
    at.add_argument("--seed", type=int, default=0)
    bn = sub.add_parser("bench", help="put YOUR detector through the Gauntlet (python function or HTTP endpoint)")
    src = bn.add_mutually_exclusive_group(required=True)
    src.add_argument("--defender", help="'module:function' or 'path/to/file.py:function' (list[str] -> list[float])")
    src.add_argument("--url", help="endpoint accepting POST {\"text\": ...} and returning {\"score\": float}")
    bn.add_argument("--name", help="label for your detector in the table")
    bn.add_argument("--per-text", action="store_true", help="your function takes ONE string instead of a list")
    bn.add_argument("--baselines", action="store_true", help="also run the built-in naive/hardened/windowed for comparison")
    bn.add_argument("--generations", type=int, default=10)
    bn.add_argument("--pop", type=int, default=150)
    bn.add_argument("--seed", type=int, default=0)
    bn.add_argument("--json", type=Path, help="write full results to this file")
    jc = sub.add_parser("judge-compare", help="compare the keyword victim oracle against a small NLI model oracle")
    jc.add_argument("--defender", default="hardened", help="defender to evolve evasions against")
    jc.add_argument("--n", type=int, default=120, help="plain attacks to sample (evasions sampled separately)")
    jc.add_argument("--seed", type=int, default=0)
    ml = sub.add_parser("multilingual-check", help="does non-English training data narrow the real-data AUC gap?")
    ml.add_argument("--defender", default="hardened")
    ml.add_argument("--frac", type=float, default=0.25, help="fraction of training benign text in German/French/Spanish")
    ml.add_argument("--seed", type=int, default=0)
    sv = sub.add_parser("serve", help="web playground (scan prompts, run the attacker); local by default, see --host")
    sv.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 inside a container")
    sv.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="default $PORT or 8000")
    sv.add_argument("--kinds", default=",".join(experiment.KINDS), help="defenders to load")
    sv.add_argument("--public", action="store_true",
                    help="public-demo limits: stricter rate limits and a smaller attack budget (also: GAUNTLET_PUBLIC=1)")
    wm = sub.add_parser("warm", help="train the linear defenders once and cache them, so scan/attack/serve start instantly")
    wm.add_argument("--kinds", default=",".join(experiment.KINDS))
    wm.add_argument("--seed", type=int, default=0)
    wm.add_argument("--clear", action="store_true", help="delete the cache instead")
    rep = sub.add_parser("report", help="re-render report.html from an existing results.json")
    rep.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)

    if args.cmd == "scan":
        texts = list(args.prompts)
        if args.file:
            texts += [ln.strip() for ln in args.file.read_text().splitlines() if ln.strip()]
        if not texts and not sys.stdin.isatty():
            texts = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
        if not texts:
            ap.error("scan: give prompts as arguments, --file, or via stdin")
        kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
        bad = [k for k in kinds if k not in experiment.ALL_KINDS]
        if bad:
            ap.error(f"unknown defender(s): {', '.join(bad)}")
        rows = scan.scan(texts, scan.train_defenders(kinds))
        print(json.dumps(rows, indent=1) if args.json else scan.render(rows))
        return
    if args.cmd == "multilingual-check":
        try:
            r = multilingual.run(args.defender, args.seed, args.frac)
        except RuntimeError as exc:
            ap.error(str(exc))
        print(multilingual.render(r))
        return
    if args.cmd == "judge-compare":
        try:
            from . import judge
        except ImportError as exc:
            ap.error(f"needs torch + transformers: pip install -r requirements-transformer.txt ({exc})")
        from .corpus import build_dataset
        df = build_dataset(args.seed)
        d = scan.train_defenders((args.defender,), seed=args.seed, df=df)[args.defender]
        j = judge.ModelJudge()
        plain = judge.compare(*judge.sample_attacks(df, args.n, args.seed), j)
        evasions = judge.compare(*judge.sample_evasions(df, d, seed=args.seed), j)
        print(judge.render(plain, evasions))
        return
    if args.cmd == "bench":
        try:
            if args.url:
                mine = bench.http_defender(args.url, args.name or "http")
            else:
                fn = bench.load_callable(args.defender)          # validates the spec with a readable error
                target, func = args.defender.rsplit(":", 1)
                mine = bench.CallableDefender(fn, args.name or Path(target).stem or func, per_text=args.per_text)
            results = []
            if args.baselines:
                for d in scan.train_defenders(seed=args.seed).values():
                    results.append(bench.run_bench(d, seed=args.seed, generations=args.generations, pop_size=args.pop))
            results.append(bench.run_bench(mine, seed=args.seed, generations=args.generations, pop_size=args.pop))
        except (ValueError, OSError) as exc:
            ap.error(str(exc))
        print(bench.render(results))
        if args.json:
            args.json.write_text(json.dumps(results, indent=1, default=float))
            print(f"\nfull results: {args.json}")
        return
    if args.cmd == "attack":
        if args.defender not in experiment.ALL_KINDS:
            ap.error(f"unknown defender: {args.defender}")
        d = scan.train_defenders((args.defender,), seed=args.seed)[args.defender]
        try:
            res = attack.attack(args.goal, d, args.generations, args.pop, args.seed)
        except ValueError as exc:
            ap.error(str(exc))
        print(attack.render(res, args.defender))
        return
    if args.cmd == "serve":
        from . import serve
        kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
        bad = [k for k in kinds if k not in experiment.ALL_KINDS]
        if bad:
            ap.error(f"unknown defender(s): {', '.join(bad)}")
        serve.run(kinds, args.port, host=args.host, public=args.public or os.environ.get("GAUNTLET_PUBLIC") == "1")
        return
    if args.cmd == "warm":
        if args.clear:
            print(f"removed {cache.clear()} cached model(s) from {cache.cache_dir()}")
            return
        kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
        bad = [k for k in kinds if k not in experiment.KINDS]
        if bad:
            ap.error(f"warm only caches the linear defenders ({', '.join(experiment.KINDS)}), not: {', '.join(bad)}")
        if not cache.enabled():
            ap.error("cache disabled by GAUNTLET_CACHE=0")
        scan.train_defenders(kinds, seed=args.seed, log=print)
        print(f"cached {', '.join(kinds)} in {cache.cache_dir()}")
        return
    if args.cmd == "run":
        if args.quick:
            args.seeds, args.rounds, args.generations, args.pop = 1, 3, 8, 120
        kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
        bad = [k for k in kinds if k not in experiment.ALL_KINDS]
        if bad:
            ap.error(f"unknown defender(s): {', '.join(bad)}")
        res = experiment.run(args.seeds, args.rounds, args.generations, args.pop, kinds=kinds)
        path = args.out / "results.json"
        if args.merge and path.exists():
            res = experiment.merge(json.loads(path.read_text()), res)
        experiment.save(res, path)
    report.write(args.out / "results.json", args.out / "report.html")
    print(f"wrote {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
