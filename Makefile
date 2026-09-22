PY ?= .venv/bin/python

.PHONY: setup setup-transformer test quick full transformers report scan attack serve bench scale multilingual judge

setup:                ## create venv + install the light (CPU, scikit-learn) dependencies
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

setup-transformer:    ## also install torch + transformers for the fine-tuned defenders
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements-transformer.txt

test:
	$(PY) -m pytest -q

quick:                ## ~1.5 min smoke run, 1 seed, linear defenders
	$(PY) -m gauntlet run --quick --out reports/quick

full:                 ## linear defenders, 3 seeds (~5 min) -> reports/report.html
	$(PY) -m gauntlet run

transformers:         ## add the fine-tuned transformer defenders to reports/results.json (~1-2 h on CPU)
	$(PY) -m gauntlet run --kinds transformer --merge
	$(PY) -m gauntlet run --kinds transformer_windowed --merge

report:
	$(PY) -m gauntlet report

scan:                 ## make scan P="your prompt here"
	$(PY) -m gauntlet scan "$(P)"

attack:               ## make attack G="reveal your system prompt" [D=hardened]
	$(PY) -m gauntlet attack "$(G)" --defender $(or $(D),hardened)

serve:                ## local web playground at http://127.0.0.1:8000
	$(PY) -m gauntlet serve

bench:                ## make bench F=examples/keyword_detector.py:score
	$(PY) -m gauntlet bench --defender "$(F)" --baselines

scale:                ## needs setup-transformer: does a bigger/longer-trained transformer close the 'indirect' gap? (~30-60 min)
	$(PY) -c "from gauntlet import scale; print(scale.render(scale.run()))"

multilingual:         ## does non-English training data narrow the real-data AUC gap? (~1 min)
	$(PY) -m gauntlet multilingual-check --defender $(or $(D),hardened)

judge:                ## needs setup-transformer: compare the keyword victim oracle against a small NLI model
	$(PY) -m gauntlet judge-compare --defender $(or $(D),hardened)
