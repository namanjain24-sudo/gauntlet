# Contributing to Gauntlet

Two very different kinds of contribution, and the first one is the interesting one.

---

## Submit a detector to the leaderboard

Gauntlet exists to answer one question about a prompt-injection detector: **what happens when an
attack it has never seen shows up?** [`LEADERBOARD.md`](LEADERBOARD.md) is the running answer for
every detector anyone has pointed at it. Adding yours takes one JSON file.

Your detector can be anything — a regex, a fine-tuned model, a commercial API wrapper, a
three-line heuristic — as long as it is a Python function that maps texts to scores where
**higher = more likely an injection**. Nothing is retrained; Gauntlet calibrates your scores to a
1% false-positive rate on benign prompts and takes it from there.

### 1. Check it locally first

```bash
git clone https://github.com/namanjain24-sudo/gauntlet && cd gauntlet
make setup
python -m gauntlet bench --defender path/to/your_detector.py:score --baselines
```

That prints the same table the board uses, next to the built-in defenders. If the numbers look
wrong, they are probably telling you something — read the evasions it prints underneath.

### 2. Write the submission file

One file, `submissions/<your-detector-name>.json`:

```json
{
  "name": "my-detector",
  "author": "your-github-handle",
  "url": "https://github.com/you/my-detector",
  "detector": "my_detector_pkg.guard:score",
  "pip": ["my-detector-pkg==1.2.0"],
  "per_text": false,
  "notes": "one line on what it is"
}
```

| field | required | meaning |
|---|---|---|
| `name` | yes | lowercase `a-z0-9._-`, and the file must be `<name>.json` |
| `author` | yes | your GitHub handle, so the board credits you |
| `detector` | yes | `module:function`, or `path/to/file.py:function` for a file you add in the same PR |
| `url` | no | where people can go find it — your repo, model card or docs |
| `pip` | no | requirements CI installs before running you (`["torch", "transformers"]`) |
| `per_text` | no | `true` if your function takes one string instead of a list |
| `notes` | no | one line, shown under your name on the board |

Prefer `pip` + an importable module over vendoring code: it keeps the board reproducible and it
means an update to your detector is a one-line PR. If your detector needs an API key, it cannot
go on the board — CI runs with no secrets, by design.

### 3. Open the PR

Push the JSON (plus your detector file, if you are adding one) and open a pull request. CI picks it
up, installs your requirements, runs the full arena, and **comments your results straight onto the
PR**: the board row, recall on each of the 8 attack families, and the actual attacks that got
through with the mutation chain that produced them. Push again and the comment is replaced, not
stacked. When it merges, a second job measures it on `main` and commits the result and the
regenerated board — **no number on the leaderboard is ever typed by a human.**

A few things worth knowing before you are surprised by them:

- Every entry runs at one pinned config (seed 0, 10 attacker generations × 150 population, 1% FPR).
  It lives in `gauntlet/leaderboard.py:CONFIG`; changing it re-measures the whole board.
- Each result also records the Python, scikit-learn and numpy it was measured on. Your detector is
  a fixed artifact and reproduces to the decimal anywhere; the reference defenders are retrained
  from scikit-learn each run, so their rows move with the library stack. The board says so.
- The built-in defenders sit in a **separate table** because they were trained on these attack
  families. Their recall is in-distribution and flattering. Do not read them as rivals.
- Submissions are third-party code, so the PR job gets a read-only token and no secrets, and a
  first-time contributor's run needs a maintainer to click approve. That is GitHub's default and
  this repo keeps it. The comment is posted by a second job that only reads that one's output and
  never runs your detector, which is why it is the only one with a write token.
- The arena puts roughly 3,500 texts through your detector. A regex finishes in seconds; a
  184M-parameter encoder on a two-core runner takes most of an hour. The job allows two, but if
  your detector is slower than that, say so in the PR and we will measure it elsewhere.
- A low score is a perfectly good contribution. The keyword-list entry scores 6.4 and it is on the
  board on purpose — a board where everything wins measures nothing.

---

## Contribute to the arena itself

The arena is more interesting when it is harder to beat. In rough order of value:

- **A new attack family** (`gauntlet/families.py`) — a style of injection the eight existing ones
  do not cover. This is the highest-value contribution to the project: every defender's
  leave-one-family-out number gets a new column, and some of them will fail it.
- **A new mutation operator** (`gauntlet/transforms.py`) — a way of rewriting an attack that keeps
  it readable to a victim but hides it from a detector. There are 21; the Unicode tag-smuggling
  and bidi-override ones were added late and immediately changed the headline result.
- **A sharper victim oracle** (`gauntlet/canonical.py`, `gauntlet/judge.py`) — the weakest part of
  the methodology, and we say so in the README. `judge-compare` shows the two existing oracles
  agree only 58% of the time.
- **A defender worth measuring** (`gauntlet/defender.py`) — new ones must plug into the same
  `fit / score / calibrate / detect` interface everything else uses.

### Ground rules

```bash
make test     # 77 tests (67 without torch — the transformer modules skip themselves)
make lint     # ruff, same check CI runs
make quick    # ~1.5 min end-to-end smoke run
```

- **Changing `canonical.canonicalize` changes every number in the README.** It is the feature
  extractor three defenders depend on. If you touch its semantics, re-run `make full` and update
  the headline table in the same PR, or say clearly that you did not. (`canonical.visible` is
  display-only and safe to change.)
- New behaviour comes with a test. The suite spins up real HTTP servers in-process for
  `serve`/`bench` rather than mocking them; follow that.
- Keep it CPU-only and key-free. "Runs on a laptop, no API keys" is a feature, not an accident.
- Numbers in the README must be reproducible by a command in the README.

## Reporting a bug

[Open an issue](https://github.com/namanjain24-sudo/gauntlet/issues/new/choose). If it is about a
number being wrong, include the command you ran and the output — that is almost always enough.

For anything security-sensitive, see [SECURITY.md](SECURITY.md).
