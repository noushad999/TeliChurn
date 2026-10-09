PY ?= python
SUBSCRIBERS ?= 300000
MONTH ?= 2025-09

.PHONY: install simulate validate train score backtest pipeline serve dashboard figures test lint docker

install:        ## install package + dev and dashboard extras
	$(PY) -m pip install -e ".[dev,dashboard]"

simulate:       ## generate the synthetic operator data warehouse
	churn simulate --subscribers $(SUBSCRIBERS)

validate:       ## check the configured data source against the data contract
	churn validate

train:          ## build snapshots, train churn + uplift models, register, gate
	churn train

score:          ## batch-score the latest month, build campaign list, check drift
	churn score

backtest:       ## realised performance + measured campaign effect (MONTH=YYYY-MM, scored earlier)
	churn score --month $(MONTH) && churn backtest --month $(MONTH)

pipeline:       ## simulate -> train -> score
	churn pipeline --subscribers $(SUBSCRIBERS)

serve:          ## run the scoring API on :8000
	churn serve

dashboard:      ## CVM dashboard on :8501
	streamlit run src/churn/dashboard.py

figures:        ## regenerate README figures from the latest run
	$(PY) docs/assets/make_figures.py

test:
	$(PY) -m pytest

lint:
	ruff check src tests && ruff format --check src tests

docker:         ## pipeline + API + dashboard + Prometheus
	docker compose up --build
