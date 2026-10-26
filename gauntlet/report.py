"""Render results.json into a single self-contained HTML report (inline SVG, no JS)."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .canonical import visible

ORDER = ("naive", "hardened", "windowed", "transformer", "transformer_windowed")   # fixed colour slot per defender
BLURB = {
    "naive": "raw text → TF-IDF → logistic regression",
    "hardened": "canonicalise (decode, de-obfuscate) first",
    "windowed": "hardened + max score over sentence windows",
    "transformer": "raw text → fine-tuned MiniLM encoder",
    "transformer_windowed": "hardened + windows, fine-tuned MiniLM",
}
W = 720


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def _bar_h(x: float, y: float, w: float, h: float, r: float = 3) -> str:
    """Horizontal bar, flat at the baseline, rounded at the data end."""
    w = max(w, 0.5)
    r = min(r, w, h / 2)
    return (f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 -{r},{r}h-{w - r:.1f}z")


def chart_lofo(g: pd.DataFrame, kinds) -> str:
    fams = sorted(g.family.unique())
    bh = 9 if len(kinds) <= 3 else 7
    block = len(kinds) * (bh + 2)
    left, right, top, gh = 128, 24, 34, block + 13
    H = top + len(fams) * gh + 34
    x = lambda v: left + v * (W - left - right)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Out-of-family recall by attack family and defender">']
    for t in (0, .25, .5, .75, 1):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{H - 28}"/>'
                   f'<text class="tick" x="{x(t):.1f}" y="{H - 10}" text-anchor="middle">{int(t * 100)}%</text>')
    for i, fam in enumerate(fams):
        y0 = top + i * gh
        out.append(f'<text class="lbl" x="{left - 10}" y="{y0 + block / 2 + 3:.1f}" text-anchor="end">{fam.replace("_", " ")}</text>')
        for kind in kinds:
            k = ORDER.index(kind)
            row = g[(g.family == fam) & (g.defender == kind)].iloc[0]
            y = y0 + kinds.index(kind) * (bh + 2)
            out.append(f'<path class="s{k}" d="{_bar_h(x(0), y, x(row.ood_recall) - x(0), bh)}">'
                       f'<title>{kind} — {fam.replace("_", " ")}: {pct(row.ood_recall, 1)} caught when the family was never seen '
                       f'(range across seeds {pct(row.ood_min)}–{pct(row.ood_max)}; in-distribution {pct(row.id_recall, 1)})</title></path>')
    out.append("</svg>")
    return "".join(out)


def chart_arms(a: pd.DataFrame, kinds) -> str:
    left, right, top, bottom = 48, 170, 16, 34
    H = 300
    rounds = sorted(a["round"].unique())
    x = lambda r: left + (r / max(rounds)) * (W - left - right)
    y = lambda v: top + (1 - v) * (H - top - bottom)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Attack success rate by round">']
    for t in (0, .25, .5, .75, 1):
        out.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                   f'<text class="tick" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{int(t * 100)}%</text>')
    for r in rounds:
        out.append(f'<text class="tick" x="{x(r):.1f}" y="{H - 12}" text-anchor="middle">round {r + 1}</text>')
    ends = []
    for kind in kinds:
        k = ORDER.index(kind)
        d = a[a.defender == kind].sort_values("round")
        pts = " ".join(f"{x(r):.1f},{y(v):.1f}" for r, v in zip(d["round"], d.asr_mean))
        out.append(f'<polyline class="ln s{k}" points="{pts}"/>')
        for _, row in d.iterrows():
            out.append(f'<circle class="dot s{k}" cx="{x(row["round"]):.1f}" cy="{y(row.asr_mean):.1f}" r="4.5">'
                       f'<title>{kind}, round {int(row["round"]) + 1}: attacker success {pct(row.asr_mean, 1)} '
                       f'(seeds {pct(row.asr_min)}–{pct(row.asr_max)}); defender recall on clean test attacks {pct(row.recall_mean, 1)}</title></circle>')
        ends.append((y(d.asr_mean.iloc[-1]), kind, k))
    placed = 1e9
    for yy, kind, k in sorted(ends, reverse=True):  # bottom-up, so labels near 0% shift up, not into the x-axis ticks
        yy = min(yy, placed - 15)
        placed = yy
        out.append(f'<rect class="s{k}" x="{W - right + 10}" y="{yy - 4:.1f}" width="8" height="8" rx="2"/>'
                   f'<text class="lbl" x="{W - right + 24}" y="{yy + 4:.1f}">{kind.replace("_", " ")}</text>')
    out.append("</svg>")
    return "".join(out)


def chart_external(e: pd.DataFrame, kinds) -> str:
    left, right, top = 150, 90, 20
    lo, hi = 0.5, 1.0
    H = top + len(kinds) * 52 + 40
    x = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Real-world AUC before and after adversarial training">']
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{H - 30}"/>'
                   f'<text class="tick" x="{x(t):.1f}" y="{H - 12}" text-anchor="middle">{t:.1f}</text>')
    for i, kind in enumerate(kinds):
        k = ORDER.index(kind)
        yy = top + 26 + i * 52
        r0 = e[(e.defender == kind) & (e.stage == "initial")].iloc[0]
        r1 = e[(e.defender == kind) & (e.stage == "final")].iloc[0]
        out.append(f'<text class="lbl" x="{left - 12}" y="{yy + 4}" text-anchor="end">{kind.replace("_", " ")}</text>'
                   f'<line class="link" x1="{x(r0.auc_mean):.1f}" x2="{x(r1.auc_mean):.1f}" y1="{yy}" y2="{yy}"/>'
                   f'<circle class="hollow s{k}" cx="{x(r0.auc_mean):.1f}" cy="{yy}" r="6"><title>{kind} before adversarial training: AUC {r0.auc_mean:.3f} (seeds {r0.auc_min:.3f}–{r0.auc_max:.3f})</title></circle>'
                   f'<circle class="dot s{k}" cx="{x(r1.auc_mean):.1f}" cy="{yy}" r="6"><title>{kind} after adversarial training: AUC {r1.auc_mean:.3f} (seeds {r1.auc_min:.3f}–{r1.auc_max:.3f})</title></circle>')
        right_pt = max(r0.auc_mean, r1.auc_mean)
        out.append(f'<text class="val" x="{x(right_pt) + 14:.1f}" y="{yy + 4}">{r0.auc_mean:.2f} → {r1.auc_mean:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def _table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def _show(text: str) -> str:
    return html.escape(visible(text))


def aggregate(res: dict):
    lo = pd.DataFrame(res["lofo"])
    g = lo.groupby(["defender", "family"]).agg(
        id_recall=("id_recall", "mean"), ood_recall=("ood_recall", "mean"),
        ood_min=("ood_recall", "min"), ood_max=("ood_recall", "max"), fpr=("fpr_test", "mean")).reset_index()
    ar = pd.DataFrame(res["arms"])
    a = ar.groupby(["defender", "round"]).agg(
        asr_mean=("asr_end", "mean"), asr_min=("asr_end", "min"), asr_max=("asr_end", "max"),
        recall_mean=("recall_test", "mean"), fpr_mean=("fpr_test", "mean"), n_ev=("n_evasions", "mean")).reset_index()
    ex = pd.DataFrame(res["external"])
    e = None
    if len(ex):
        e = ex.groupby(["defender", "stage"]).agg(
            auc_mean=("auc", "mean"), auc_min=("auc", "min"), auc_max=("auc", "max"),
            recall=("recall", "mean"), fpr=("fpr", "mean")).reset_index()
    return g, a, e


def findings(g, a, e, kinds) -> list[str]:
    out = []
    for kind in kinds:
        d = g[g.defender == kind]
        worst = d.sort_values("ood_recall").iloc[0]
        name = kind.replace("_", " ")
        out.append(f"<b>{name}</b>: mean recall on a never-seen attack family is {pct(d.ood_recall.mean(), 1)} "
                   f"(in-distribution {pct(d.id_recall.mean(), 1)}); weakest family is "
                   f"<i>{worst.family.replace('_', ' ')}</i> at {pct(worst.ood_recall)}.")
    for kind in kinds:
        d = a[a.defender == kind].sort_values("round")
        name = kind.replace("_", " ")
        out.append(f"<b>{name}</b> under attack: evolved attackers evade it {pct(d.asr_mean.iloc[0])} of the time in round 1 "
                   f"and {pct(d.asr_mean.iloc[-1])} in round {len(d)} (peak later round: {pct(d.asr_mean.iloc[1:].max())}).")
    if e is not None:
        for kind in kinds:
            r0 = e[(e.defender == kind) & (e.stage == "initial")].iloc[0]
            r1 = e[(e.defender == kind) & (e.stage == "final")].iloc[0]
            name = kind.replace("_", " ")
            out.append(f"<b>{name}</b> on real human-written injections: AUC {r0.auc_mean:.2f} before adversarial training, "
                       f"{r1.auc_mean:.2f} after.")
    return out


REPO = "https://github.com/namanjain24-sudo/gauntlet"

CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--s3:#eda100;--s4:#e87ba4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#c98500;--s4:#d55181}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#c98500;--s4:#d55181}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 20px 80px}h1{font-size:2rem;margin:0 0 6px;letter-spacing:-.02em}h2{font-size:1.15rem;margin:0 0 4px}
.sub{color:var(--ink2);margin:0 0 28px}.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 20px 16px;margin:0 0 20px}
.card p{margin:4px 0 12px;color:var(--ink2)}svg{width:100%;height:auto;display:block}.grid{stroke:var(--grid);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}.lbl{fill:var(--ink);font-size:12.5px}.val{fill:var(--ink2);font-size:12px}
.s0{fill:var(--s0);stroke:var(--s0)}.s1{fill:var(--s1);stroke:var(--s1)}.s2{fill:var(--s2);stroke:var(--s2)}.s3{fill:var(--s3);stroke:var(--s3)}.s4{fill:var(--s4);stroke:var(--s4)}
path.s0,path.s1,path.s2,path.s3,path.s4,rect.s0,rect.s1,rect.s2,rect.s3,rect.s4{stroke:none}
.ln{fill:none;stroke-width:2}.dot{stroke:var(--surface);stroke-width:2}.hollow{fill:var(--surface);stroke-width:2.5}.link{stroke:var(--axis);stroke-width:2}
a{color:var(--s0);text-decoration:none;border-bottom:1px solid rgba(127,127,127,.35)}a:hover{border-bottom-color:currentColor}
.home{margin:0 0 28px;font-size:13.5px;color:var(--muted)}.home a{margin-right:4px}.home span{margin:0 8px}
footer{margin-top:36px;padding-top:18px;border-top:1px solid var(--border);color:var(--muted);font-size:13.5px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 10px;font-size:13px;color:var(--ink)}.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.legend small{color:var(--ink2)}ul{padding-left:18px;margin:8px 0}li{margin:5px 0}details{margin-top:8px}summary{cursor:pointer;color:var(--ink2);font-size:13px}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid)}th{color:var(--ink2);font-weight:600}
code,.mono{font:12.5px ui-monospace,Menlo,monospace}.atk{border-left:3px solid var(--axis);padding:4px 10px;margin:8px 0;background:transparent;word-break:break-word}
.tag{color:var(--ink2);font-size:12px}.note{font-size:13px;color:var(--ink2)}
"""


def render(res: dict) -> str:
    g, a, e = aggregate(res)
    cfg = res["config"]
    present = set(g.defender)
    kinds = [k for k in ORDER if k in present]
    legend = '<div class="legend">' + "".join(
        f'<span><i class="s{ORDER.index(kind)}" style="background:var(--s{ORDER.index(kind)})"></i>{kind.replace("_", " ")} <small>— {BLURB[kind]}</small></span>'
        for kind in kinds) + "</div>"
    lofo_tbl = _table(g.assign(id_recall=g.id_recall.map(lambda v: pct(v, 1)), ood_recall=g.ood_recall.map(lambda v: pct(v, 1)),
                                fpr=g.fpr.map(lambda v: pct(v, 1)))[["defender", "family", "id_recall", "ood_recall", "fpr"]]
                      .rename(columns={"id_recall": "seen in training", "ood_recall": "never seen", "fpr": "benign FPR (test)"}))
    arms_tbl = _table(a.assign(**{c: a[c].map(lambda v: pct(v, 1)) for c in ("asr_mean", "asr_min", "asr_max", "recall_mean", "fpr_mean")},
                               n_ev=a.n_ev.round(0).astype(int), round=a["round"] + 1)
                      .rename(columns={"asr_mean": "attacker success", "asr_min": "min", "asr_max": "max",
                                       "recall_mean": "recall (clean)", "fpr_mean": "FPR", "n_ev": "evasions found"}))
    ext_block = ""
    if e is not None:
        ext_tbl = _table(e.assign(auc_mean=e.auc_mean.round(3), auc_min=e.auc_min.round(3), auc_max=e.auc_max.round(3),
                                  recall=e.recall.map(lambda v: pct(v, 1)), fpr=e.fpr.map(lambda v: pct(v, 1)))
                         .rename(columns={"auc_mean": "AUC", "auc_min": "min", "auc_max": "max"}))
        ext_block = f"""<section class="card"><h2>3 · Does patching the attacker help in the real world?</h2>
<p>Public human-written injections (<code>deepset/prompt-injections</code>, n={cfg['external_n']}, English + German). Threshold-free AUC, before vs after each defender was adversarially trained on the attacker's evasions.</p>
<div class="legend"><span>○ before adversarial training</span><span>● after</span></div>{chart_external(e, kinds)}
<details><summary>Data table</summary>{ext_tbl}</details></section>"""
    gal = ""
    for kind in kinds:
        items = res["gallery"].get(kind, [])
        if items:
            gal += f"<h2 style='margin-top:14px'>{kind.replace('_', ' ')}</h2>" + "".join(
                f'<div class="atk mono">{_show(i["text"])}<div class="tag">evolved via: {" → ".join(i["lineage"])}</div></div>' for i in items[:3])
    mut = "".join(f"<li><b>{k.replace('_', ' ')}</b>: " + ", ".join(f"{m} ({n})" for m, n in list(res["mutators"][k].items())[:6]) + "</li>" for k in kinds)
    fnd = "".join(f"<li>{s}</li>" for s in findings(g, a, e, kinds))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gauntlet report</title><style>{CSS}</style></head><body><main>
<h1>Gauntlet</h1><p class="sub">Adversarial co-evolution arena for prompt-injection defenses · {cfg['seeds']} seeds · {cfg['rounds']} rounds × {cfg['generations']} generations · population {cfg['pop_size']}</p>
<p class="home"><a href="{REPO}">source on GitHub</a><span>·</span><a href="{REPO}/blob/main/LEADERBOARD.md">leaderboard</a><span>·</span><a href="{REPO}/blob/main/CONTRIBUTING.md#submit-a-detector-to-the-leaderboard">put your own detector through this</a></p>
<section class="card"><h2>What the numbers say</h2><ul>{fnd}</ul></section>
<section class="card"><h2>1 · Can it catch an attack family it has never seen?</h2>
<p>Leave-one-family-out: the defender is retrained without one family, then tested on it. Threshold calibrated to 1% false positives on benign validation prompts. Bars show the mean over seeds; hover for ranges.</p>
{legend}{chart_lofo(g, kinds)}<details><summary>Data table</summary>{lofo_tbl}</details></section>
<section class="card"><h2>2 · Arms race: an evolving attacker vs a retraining defender</h2>
<p>Each round the attacker mutates payloads for {cfg['generations']} generations against the current defender; the defender then retrains on what got through. Every round uses fresh malicious goals, so later rounds test whether patching generalised. Attacker success = share of the final population that evades detection <i>and</i> is still understood by the victim oracle.</p>
{legend}{chart_arms(a, kinds)}<details><summary>Data table</summary>{arms_tbl}</details></section>
{ext_block}
<section class="card"><h2>What the attacker learned</h2><p>Mutation operators most often present in successful evasions (count of distinct evasions, summed over seeds).</p><ul>{mut}</ul>
<details><summary>Example evolved attacks (round 1, seed 0)</summary>{gal}</details></section>
<section class="card"><h2>Limitations (read these)</h2><ul>
<li>Attacks and benign prompts are <b>synthetic</b>, built from templates. Malicious goals are split by string between train/test/arena, but they share a vocabulary. High in-distribution numbers say little; the out-of-family and real-data numbers are the honest ones.</li>
<li>The <b>victim oracle</b> is a keyword-recovery proxy, not a real LLM. It can accept mutants a real model would not follow (e.g. heavy interleaving) and reject some it would.</li>
<li>Thresholds are calibrated on synthetic benign text. On real data the operating point does not transfer, which is why AUC is reported alongside recall/FPR.</li>
<li>Three defenders are linear models over TF-IDF; two are one small fine-tuned transformer (MiniLM-L6, 160 optimiser steps per fit, CPU budget). A larger model or a longer schedule may behave differently, so read the transformer rows as "a small fine-tuned encoder does not close the gap", not as a statement about all transformers.</li>
</ul></section>
<footer>Every number above is reproducible with <code>make full</code> from <a href="{REPO}">the repository</a>, on a laptop CPU, with no API keys. Bring your own detector: <a href="{REPO}/blob/main/CONTRIBUTING.md#submit-a-detector-to-the-leaderboard">it takes one JSON file</a>.</footer>
</main></body></html>"""


def write(results_path: Path, out_path: Path) -> None:
    res = json.loads(results_path.read_text())
    out_path.write_text(render(res), encoding="utf-8")
