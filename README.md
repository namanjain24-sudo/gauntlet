# Gauntlet

**An adversarial arena that measures whether a prompt-injection defense generalises to attacks it has never seen.**

Most detectors report "99% accuracy" on a test set drawn from the same distribution they trained on.
The question a company actually has is: *what happens when a new kind of attack shows up?*
Gauntlet answers that with three experiments, all CPU-only, no API keys:

| # | Experiment | What it measures |
|---|------------|------------------|
| 1 | **Leave-one-family-out** over 8 attack families | Recall on a family the defender was never trained on, vs. when it was (bootstrap-able, threshold fixed at 1% benign FPR) |
| 2 | **Evolutionary attacker vs. retraining defender** | Attacker mutates payloads to evade the current defender; the defender retrains on what got through. Each round uses *fresh* malicious goals, so it tests whether patching generalises |
| 3 | **Sim-to-real check** on human-written injections ([deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)) | Threshold-free AUC before/after adversarial training |

An evolved attack only counts if a **victim oracle** can still recover the malicious intent, so the attacker cannot "win" by destroying the payload.

Three defenders are compared: `naive` (TF-IDF + logistic regression), `hardened` (canonicalise first: strip invisible chars, fold homoglyphs, de-leet, decode base64/hex/ROT13), and `windowed` (hardened, but scores every sentence window and takes the max).

![report preview](reports/preview.png)

## Results (3 seeds; full report in `reports/report.html`)

**1. Unseen attack families.** In-distribution recall is ~100% for every defender, which is exactly why in-distribution numbers are uninformative.

| defender | mean recall on unseen family | weakest family |
|---|---|---|
| naive | 86.6% | indirect (docs with hidden instructions): 30% |
| hardened | 96.7% | indirect: 74% |
| windowed | 96.5% | indirect: 89% |

Canonicalisation alone lifts the unseen-`encoding` family from ~64% to ~99%. Indirect injection is the hard one for everything.

**2. Arms race.** Attacker success rate (evasion that the victim oracle still understands):

| defender | round 1 | round 5 | worst later round (mean) |
|---|---|---|---|
| naive | 75% | 1% | 6% |
| hardened | 91% | 0% | 25% (one seed: 50%) |
| windowed | 46% | 27% | 36% |

`naive` and `hardened` are "patched" to ~0% after one round of adversarial training, but `hardened` reopens in round 4 when the attacker finds a new trick. `windowed` never reaches zero.

**3. Real human-written injections (AUC).**

| defender | before adversarial training | after |
|---|---|---|
| naive | 0.57 | 0.60 |
| hardened | 0.67 | 0.71 |
| windowed | 0.81 | 0.81 |

**Takeaway:** an attack success rate near 0 inside the arena says little about real-world quality. The defenders that look "solved" by adversarial training are the ones that transfer worst (AUC 0.60 / 0.71), while the one the attacker keeps breaking has the best real-world ranking (0.81). Architecture (canonicalise, score windows) moved the real-data needle far more than data patches did (+0.03 to +0.04).

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                 # 21 tests
python -m gauntlet run --quick      # ~1 min, 1 seed
python -m gauntlet run              # full: 3 seeds, ~5 min -> reports/report.html
python -m gauntlet report           # re-render the HTML from reports/results.json
```

## Layout

```
gauntlet/
  transforms.py   16 text mutation operators (leetspeak, homoglyphs, base64, padding, interleaving, ...)
  families.py     8 attack families (direct, roleplay, authority, encoding, payload split, multilingual, indirect, obfuscation)
  corpus.py       synthetic malicious goals + benign prompts, incl. hard negatives that reuse attack vocabulary
  canonical.py    canonicalisation + the victim oracle
  defender.py     naive / hardened / windowed defenders, threshold calibrated on benign validation data
  evaluate.py     leave-one-family-out with bootstrap CIs
  evolve.py       evolutionary attacker + arms-race loop
  external.py     real-data check (downloads and caches deepset/prompt-injections)
  experiment.py   multi-seed runner      report.py   self-contained HTML (inline SVG, dark mode)
```

## Design decisions worth knowing

- **Goals are split before rendering** (train/test/arena are disjoint strings), so memorising payloads cannot pass.
- **Hard negatives share the attack vocabulary** ("delete all rows in SQL", "send the report to my manager"). An earlier version without them scored ~100% everywhere because the model was just spotting vocabulary; adding them exposed the real weaknesses.
- **Thresholds are calibrated on benign validation data only**, never on attacks.

## Limitations (please read)

- Attacks and benign prompts are **synthetic and template-built**. The out-of-family and real-data numbers are the trustworthy ones.
- The **victim oracle is a keyword-recovery proxy**, not a real LLM. It may accept mutants a real model would not follow (e.g. heavy interleaving).
- Calibration does not transfer: 1% FPR on validation became ~3% on the test split for naive/hardened, and on real data `windowed` catches only ~20-35% at its threshold despite the best AUC.
- The external set is English + German while training data is English only; part of the AUC gap may be language, not attack style.
- Linear models over TF-IDF. Whether the same failure pattern holds for a fine-tuned transformer is untested, and is the obvious next step.
