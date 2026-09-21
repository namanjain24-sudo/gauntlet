PY ?= .venv/bin/python

.PHONY: setup setup-transformer test quick full transformers report scan

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
