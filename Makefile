SHELL := /bin/bash

PYTHON ?= var/seed_paper_eval/paper310/bin/python
PYTHON_LIB ?= var/seed_paper_eval/paper310/lib
PLOT_PYTHON ?= python
export LD_LIBRARY_PATH := $(abspath $(PYTHON_LIB))$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))

.PHONY: help check compile lint test shell figures freeze-3b-result e21-figure e22-figures e22-v2-config current-canonical-figure e44-figure math-divergence-figure paper dry-run e11-config e16-config e18-config e19-config monitor

help:
	@printf '%s\n' \
	  'make check    Compile, lint, test, and syntax-check the maintained code' \
	  'make figures  Refresh all MaxEnt paper and live compute-divergence figures, including E21/E22' \
	  'make freeze-3b-result  Freeze the latest valid paired 3B common-horizon result' \
	  'make e21-figure  Refresh both live free-form MATH MaxEnt figures' \
	  'make e22-figures  Refresh E22 curves and all Countdown/coloring live figures' \
	  'make e22-v2-config  Validate the matched base-preserving E22-v2 cohort without submitting' \
	  'make math-divergence-figure  Refresh the compute-divergence-style MATH figure' \
	  'make paper    Rebuild paper/main.pdf from the single manuscript source' \
	  'make dry-run  Print the resolved standard-MaxEnt training command' \
	  'make monitor  Watch E51 with matched Dr.GRPO and refresh its live figure every minute' \
	  'make current-canonical-figure  Refresh the E51 live figure once' \
	  'make e11-config  Inspect all six standard-MaxEnt cells without submitting jobs' \
	  'make e16-config  Exhaustively inspect the E15-derived canonical smoke without creating artifacts or jobs' \
	  'make e18-config  Inspect the matched 3B canonical Dr.GRPO controls without submitting jobs' \
	  'make e19-config  Inspect the matched 0.5B canonical Dr.GRPO controls without submitting jobs'

check: compile lint test shell

compile:
	$(PYTHON) -m py_compile $$(rg --files src ops tests -g '*.py')

lint:
	$(PYTHON) -m ruff check src tests ops

test:
	$(PYTHON) -m pytest -q

shell:
	bash -n $$(rg --files ops -g '*.sh')

figures:
	$(PLOT_PYTHON) ops/exp_scaling/refresh_campaign_curves.py
	$(PLOT_PYTHON) ops/plot_task_examples.py
	$(PLOT_PYTHON) ops/plot_canonical_maxent_mechanism.py
	$(PLOT_PYTHON) ops/plot_canonical_maxent_paper.py
	$(PLOT_PYTHON) ops/exp_scaling/plot_divergence.py
	$(PLOT_PYTHON) ops/plot_e21_math_token_maxent_live.py

freeze-3b-result:
	$(PYTHON) ops/exp_scaling/freeze_canonical_maxent_3b_interim.py

e21-figure:
	$(PLOT_PYTHON) ops/plot_e21_math_token_maxent_live.py

e22-figures:
	$(PLOT_PYTHON) ops/exp_scaling/refresh_campaign_curves.py
	$(PLOT_PYTHON) ops/exp_scaling/plot_divergence.py

e22-v2-config:
	bash ops/exp_scaling/launch_e22_modebench_freeform_token_maxent_v2.sh config

current-canonical-figure:
	$(PLOT_PYTHON) ops/exp_scaling/refresh_latest_freeform_05b.py --current-canonical-only

e44-figure: current-canonical-figure

math-divergence-figure:
	$(PLOT_PYTHON) ops/plot_e21_math_token_maxent_live.py

paper:
	$(MAKE) -C paper clean all

dry-run:
	OAT_ZERO_DRY_RUN=1 OAT_ZERO_VARIANT=maxent bash ops/run_experiment.sh

monitor:
	python3 ops/exp_scaling/monitor_campaign.py --current-canonical-only --figure-refresh-seconds 60

e11-config:
	@OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1 \
	  bash ops/exp_scaling/launch_on_policy_maxent_extension.sh all

e16-config:
	@bash ops/exp_scaling/launch_e16_canonical_maxent_replication.sh smoke-config

e18-config:
	@bash ops/exp_scaling/launch_e18_canonical_drgrpo_3b_control.sh config

e19-config:
	@bash ops/exp_scaling/launch_e19_canonical_drgrpo_05b_control.sh config
