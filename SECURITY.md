# Security

## What Gauntlet is, in security terms

Gauntlet is a **measurement instrument for prompt-injection detectors**, not a detector. Its
built-in defenders are deliberately small, trained on synthetic data, and documented as such.
Do not put them in front of a production LLM and consider the problem solved — the README's
[Limitations](README.md#limitations) section exists for exactly this reason.

It does contain working prompt-injection attack generators. That is the point: they are there so
defenses can be measured against them, the same way a fuzzer ships with inputs that crash things.
They target a local synthetic corpus and your own detector, not anyone else's system.

## Running the playground in public

`python -m gauntlet serve` binds to `127.0.0.1` and is meant for your machine. If you host it:

- Use `--public` (or `GAUNTLET_PUBLIC=1`), which the Docker image sets. It turns on per-client
  rate limits, a capped attack budget, a single-attack queue, a 200 KB request cap, and
  conservative security headers.
- Set `GAUNTLET_TRUST_PROXY=1` **only** behind a reverse proxy you control. Without that, a client
  can forge `X-Forwarded-For` and walk around the rate limiter.
- There is no authentication and no persistence by design. Do not paste anything into a hosted
  playground that you would not post publicly.

## Running leaderboard submissions

Submitted detectors are arbitrary third-party code. The PR workflow runs them with a read-only
token, no secrets, and GitHub's default "first-time contributor needs approval" gate.
Maintainers: read a submission's `pip` list and detector source before approving the run.

## Reporting a vulnerability

Please **do not** open a public issue for a vulnerability in Gauntlet itself.

Use [GitHub's private vulnerability reporting](https://github.com/namanjain24-sudo/gauntlet/security/advisories/new)
on this repository. Expect a first response within a week.

Findings about *prompt injection in general*, or about a detector that scores badly on the
leaderboard, are not vulnerabilities in this project — those are results, and they belong in an
issue or a submission.
