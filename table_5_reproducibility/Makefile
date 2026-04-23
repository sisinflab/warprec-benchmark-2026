# Makefile — Reproducibility driver for Table 5
# ================================================================
#
# End-to-end reproduction of Table 5 (cross-framework evaluation of ItemKNN and
# LightGCN on MovieLens-1M) across Elliot, RecBole, and WarpRec.
#
#   make help            # list targets
#   make all             # full pipeline (~30-60 min CPU, <10 min GPU)
#   make install MODE=gpu
#   make clean           # remove reproduced artefacts (keeps envs)
#
# Three isolated micromamba environments are created; each framework is
# invoked inside its own env via `micromamba run -n <env>`, so the user does
# not need to activate anything manually.
#
# Variables (override on the command line):
#   MODE   = gpu | cpu                    [default: gpu]
#   MAMBA  = micromamba | mamba | conda   [default: first found on PATH]
# ================================================================

MODE   ?= gpu
# Auto-detect an env manager. Order of preference:
#   1. $MAMBA_EXE (exported by `micromamba shell init`)
#   2. micromamba / mamba / conda on PATH
#   3. common binary locations (micromamba's .bashrc hook does not run in
#      make's non-interactive subshell, so PATH may not include ~/.local/bin)
MAMBA  ?= $(shell \
	if [ -n "$$MAMBA_EXE" ] && [ -x "$$MAMBA_EXE" ]; then echo "$$MAMBA_EXE"; exit 0; fi; \
	for c in micromamba mamba conda; do \
		command -v $$c >/dev/null 2>&1 && { echo $$c; exit 0; }; \
	done; \
	for p in $$HOME/.local/bin/micromamba $$HOME/bin/micromamba \
	         $$HOME/micromamba/bin/micromamba /opt/micromamba/bin/micromamba \
	         /usr/local/bin/micromamba; do \
		[ -x "$$p" ] && { echo "$$p"; exit 0; }; \
	done)
PY     ?= python3

# Colors — resolve through tput when stdout is a terminal; fall back to
# empty strings in pipes / CI so the help output stays plain.
BOLD   := $(shell tput bold    2>/dev/null)
DIM    := $(shell tput dim     2>/dev/null)
RESET  := $(shell tput sgr0    2>/dev/null)
CYAN   := $(shell tput setaf 6 2>/dev/null)
GREEN  := $(shell tput setaf 2 2>/dev/null)
YELLOW := $(shell tput setaf 3 2>/dev/null)
MAGENTA:= $(shell tput setaf 5 2>/dev/null)

# Env names (stable across MODE — only the YAML used to create them changes).
ENV_ELLIOT  := table_5_reproducibility_elliot
ENV_RECBOLE := table_5_reproducibility_recbole
ENV_WARPREC := table_5_reproducibility_warprec

ENV_YML_ELLIOT  := environments/elliot_$(MODE).yml
ENV_YML_RECBOLE := environments/recbole_$(MODE).yml
ENV_YML_WARPREC := environments/warprec_$(MODE).yml

# Run-inside-env shortcuts.
RUN_ELLIOT  = $(MAMBA) run -n $(ENV_ELLIOT)
RUN_RECBOLE = $(MAMBA) run -n $(ENV_RECBOLE)
RUN_WARPREC = $(MAMBA) run -n $(ENV_WARPREC)

ELLIOT_DIR  := frameworks/elliot
RECBOLE_DIR := frameworks/recbole
WARPREC_DIR := frameworks/warprec

.PHONY: help all install install-elliot install-recbole install-warprec \
        _check-mamba \
        download train-warprec train-elliot train-recbole \
        cross-eval cross-eval-elliot cross-eval-recbole cross-eval-warprec \
        stage-recbole-inputs aggregate \
        clean clean-envs

# ---------------------------------------------------------------
help:
	@printf '\n  $(BOLD)WarpRec · Table 5 reproducibility$(RESET)\n\n'
	@printf '  Regenerates Table 5 (ItemKNN & LightGCN on MovieLens-1M across\n'
	@printf '  Elliot, RecBole, WarpRec). One command, end to end:\n\n'
	@printf '      $(GREEN)make all MODE=gpu$(RESET)   $(DIM)# NVIDIA GPU with CUDA 12.6$(RESET)\n'
	@printf '      $(GREEN)make all MODE=cpu$(RESET)   $(DIM)# no GPU required (slower)$(RESET)\n\n'
	@printf '  Output: $(CYAN)table.md$(RESET) and $(CYAN)table.tex$(RESET) (the paper file, overwritten).\n\n'
	@printf '  If a step fails, rerun the same command — it resumes from where\n'
	@printf '  it stopped. See $(CYAN)README.md$(RESET) for prerequisites and troubleshooting.\n\n'
	@printf '  $(DIM)Individual targets (advanced):$(RESET)\n'
	@printf '    $(DIM)install       create all 3 micromamba envs$(RESET)\n'
	@printf '    $(DIM)install-<fw>  create one env (warprec | elliot | recbole)$(RESET)\n'
	@printf '    $(DIM)download      fetch MovieLens-1M$(RESET)\n'
	@printf '    $(DIM)train-<fw>    train one framework (warprec | elliot | recbole)$(RESET)\n'
	@printf '    $(DIM)cross-eval    run the 6 cross-framework evaluations$(RESET)\n'
	@printf '    $(DIM)aggregate     build table.md and table.tex$(RESET)\n'
	@printf '    $(DIM)clean         remove generated artefacts$(RESET)\n'
	@printf '    $(DIM)clean-envs    remove the 3 micromamba envs$(RESET)\n\n'
	@printf '  $(DIM)Variables (override on the command line):$(RESET)\n'
	@printf '    $(DIM)MODE   = gpu | cpu                    [default: gpu]$(RESET)\n'
	@printf '    $(DIM)MAMBA  = micromamba | mamba | conda   [default: first found on PATH]$(RESET)\n'
	@printf '    $(DIM)                                       currently: $(RESET)$(CYAN)$(MAMBA)$(RESET)\n\n'

# ---------------------------------------------------------------
# Env creation
# ---------------------------------------------------------------
install: install-elliot install-recbole install-warprec

_check-mamba:
	@if [ -z "$(MAMBA)" ]; then \
		echo "error: no env manager found (looked for micromamba, mamba, conda on PATH" >&2; \
		echo "       and in common install locations; \$$MAMBA_EXE is also empty)." >&2; \
		echo "       install micromamba (https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)" >&2; \
		echo "       or override with:  make install MAMBA=/path/to/binary" >&2; \
		exit 1; \
	fi

# Elliot installs in 3 serial phases because torch_scatter / torch_sparse
# are source-built and need `import torch` to succeed at build time — so
# torch must land in its own pip call, before the PyG ecosystem. Versions
# below track $(ENV_YML_ELLIOT) (keep in sync if the YAML changes).
# --prune is intentionally omitted: torch & PyG are installed outside the
# YAML, and --prune would delete them on every re-run.
install-elliot: _check-mamba
	@echo ">>> [elliot 1/3] base env from $(ENV_YML_ELLIOT) (using $(MAMBA))"
	@$(MAMBA) env list | grep -q " $(ENV_ELLIOT) " \
		&& $(MAMBA) env update -n $(ENV_ELLIOT) -f $(ENV_YML_ELLIOT) \
		|| $(MAMBA) env create -n $(ENV_ELLIOT) -f $(ENV_YML_ELLIOT)
	@echo ">>> [elliot 2/3] torch (required before torch_scatter / torch_sparse)"
	$(RUN_ELLIOT) pip install torch==2.3.1
	@echo ">>> [elliot 3/3] PyG ecosystem + Elliot upstream"
	$(RUN_ELLIOT) pip install \
		torch_scatter==2.1.2 \
		torch_sparse==0.6.18 \
		torch-geometric==2.6.1 \
		git+https://github.com/sisinflab/elliot.git

install-recbole: _check-mamba
	@echo ">>> creating/updating $(ENV_RECBOLE) from $(ENV_YML_RECBOLE) (using $(MAMBA))"
	@$(MAMBA) env list | grep -q " $(ENV_RECBOLE) " \
		&& $(MAMBA) env update -n $(ENV_RECBOLE) -f $(ENV_YML_RECBOLE) --prune \
		|| $(MAMBA) env create -n $(ENV_RECBOLE) -f $(ENV_YML_RECBOLE)

install-warprec: _check-mamba
	@echo ">>> creating/updating $(ENV_WARPREC) from $(ENV_YML_WARPREC) (using $(MAMBA))"
	@$(MAMBA) env list | grep -q " $(ENV_WARPREC) " \
		&& $(MAMBA) env update -n $(ENV_WARPREC) -f $(ENV_YML_WARPREC) --prune \
		|| $(MAMBA) env create -n $(ENV_WARPREC) -f $(ENV_YML_WARPREC)

# ---------------------------------------------------------------
# Stage 1: data download (uses host Python; no framework dependency).
# ---------------------------------------------------------------
download: data/raw/movielens.csv

data/raw/movielens.csv:
	$(PY) scripts/download_ml1m.py

# ---------------------------------------------------------------
# Stage 2: WarpRec training + canonical split export.
# WarpRec emits the temporal-holdout split consumed by Elliot and RecBole.
# ---------------------------------------------------------------
train-warprec: data/split/train.tsv recs/warprec/ItemKNN.tsv recs/warprec/LightGCN.tsv

data/split/train.tsv data/split/test.tsv \
recs/warprec/ItemKNN.tsv recs/warprec/LightGCN.tsv: data/raw/movielens.csv
	@echo ">>> [warprec] training ItemKNN + LightGCN"
	$(RUN_WARPREC) warprec -c $(WARPREC_DIR)/train.yml -p train
	$(RUN_WARPREC) $(PY) scripts/export_split.py
	$(RUN_WARPREC) $(PY) scripts/collect_recs.py warprec

# ---------------------------------------------------------------
# Stage 3: Elliot training using the canonical split.
# ---------------------------------------------------------------
train-elliot: recs/elliot/ItemKNN.tsv recs/elliot/LightGCN.tsv

recs/elliot/ItemKNN.tsv recs/elliot/LightGCN.tsv: data/split/train.tsv
	@echo ">>> [elliot] training ItemKNN + LightGCN"
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py $(ELLIOT_DIR)/train.yml
	$(RUN_ELLIOT) $(PY) scripts/collect_recs.py elliot

# ---------------------------------------------------------------
# Stage 4: RecBole training using the canonical split.
# RecBole's run_experiment.py resolves paths relative to its own directory, so
# we stage the split into `frameworks/recbole/splits/` before invoking.
# ---------------------------------------------------------------
train-recbole: recs/recbole/ItemKNN.tsv recs/recbole/LightGCN.tsv

recs/recbole/ItemKNN.tsv recs/recbole/LightGCN.tsv: data/split/train.tsv
	@echo ">>> [recbole] staging splits and training"
	mkdir -p $(RECBOLE_DIR)/splits
	cp data/split/train.tsv $(RECBOLE_DIR)/splits/train.tsv
	cp data/split/test.tsv  $(RECBOLE_DIR)/splits/test.tsv
	cd $(RECBOLE_DIR) && $(RUN_RECBOLE) $(PY) run_experiment.py
	$(RUN_RECBOLE) $(PY) scripts/collect_recs.py recbole

# ---------------------------------------------------------------
# Stage 5: Cross-framework evaluation (6 runs; 3 native self-evals already
# produced during training, so the aggregator only needs these six extras).
# ---------------------------------------------------------------
cross-eval: cross-eval-elliot cross-eval-recbole cross-eval-warprec

# Elliot evaluates RecBole + WarpRec recs (each model gets its own YAML because
# Elliot's ProxyRecommender only supports a single path per run).
cross-eval-elliot: recs/recbole/LightGCN.tsv recs/warprec/LightGCN.tsv
	@echo ">>> [elliot] cross-evaluating RecBole + WarpRec recommendations"
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py $(ELLIOT_DIR)/eval_recbole_itemknn.yml
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py $(ELLIOT_DIR)/eval_recbole_lightgcn.yml
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py $(ELLIOT_DIR)/eval_warprec_itemknn.yml
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py $(ELLIOT_DIR)/eval_warprec_lightgcn.yml
	$(RUN_ELLIOT) $(PY) scripts/collect_metrics.py elliot elliot
	$(RUN_ELLIOT) $(PY) scripts/collect_metrics.py recbole elliot
	$(RUN_ELLIOT) $(PY) scripts/collect_metrics.py warprec elliot

# RecBole's evaluate_external_recs.py expects `recs/<framework>/<model>.tsv`
# (relative to RecBole's CWD); we stage our canonical recs into that layout.
stage-recbole-inputs: recs/elliot/LightGCN.tsv recs/warprec/LightGCN.tsv
	mkdir -p $(RECBOLE_DIR)/recs/elliot $(RECBOLE_DIR)/recs/warprec
	cp recs/elliot/ItemKNN.tsv    $(RECBOLE_DIR)/recs/elliot/ItemKNN.tsv
	cp recs/elliot/LightGCN.tsv   $(RECBOLE_DIR)/recs/elliot/LightGCN.tsv
	cp recs/warprec/ItemKNN.tsv   $(RECBOLE_DIR)/recs/warprec/ItemKNN.tsv
	cp recs/warprec/LightGCN.tsv  $(RECBOLE_DIR)/recs/warprec/LightGCN.tsv

cross-eval-recbole: stage-recbole-inputs
	@echo ">>> [recbole] cross-evaluating Elliot + WarpRec recommendations"
	cd $(RECBOLE_DIR) && $(RUN_RECBOLE) $(PY) evaluate_external_recs.py
	$(RUN_RECBOLE) $(PY) scripts/collect_metrics.py recbole recbole
	$(RUN_RECBOLE) $(PY) scripts/collect_metrics.py elliot recbole
	$(RUN_RECBOLE) $(PY) scripts/collect_metrics.py warprec recbole

cross-eval-warprec: recs/elliot/LightGCN.tsv recs/recbole/LightGCN.tsv
	@echo ">>> [warprec] cross-evaluating Elliot + RecBole recommendations"
	$(RUN_WARPREC) warprec -c $(WARPREC_DIR)/eval_elliot_itemknn.yml  -p eval
	$(RUN_WARPREC) warprec -c $(WARPREC_DIR)/eval_elliot_lightgcn.yml -p eval
	$(RUN_WARPREC) warprec -c $(WARPREC_DIR)/eval_recbole_itemknn.yml  -p eval
	$(RUN_WARPREC) warprec -c $(WARPREC_DIR)/eval_recbole_lightgcn.yml -p eval
	$(RUN_WARPREC) $(PY) scripts/collect_metrics.py warprec warprec
	$(RUN_WARPREC) $(PY) scripts/collect_metrics.py elliot  warprec
	$(RUN_WARPREC) $(PY) scripts/collect_metrics.py recbole warprec

# ---------------------------------------------------------------
# Stage 6: aggregate the 9 canonical CSVs into table.md + table.tex.
# Runs on the host interpreter (stdlib-only).
# ---------------------------------------------------------------
aggregate:
	$(PY) scripts/aggregate_results.py

# ---------------------------------------------------------------
# Full pipeline.
# ---------------------------------------------------------------
all: install download train-warprec train-elliot train-recbole cross-eval aggregate
	@echo ">>> done. See table.md and the regenerated table.tex."

# ---------------------------------------------------------------
# Cleanup.
# ---------------------------------------------------------------
clean:
	rm -rf data/ recs/ metrics/ experiment/ results/ table.md
	rm -rf $(RECBOLE_DIR)/splits $(RECBOLE_DIR)/recs \
	       $(RECBOLE_DIR)/results $(RECBOLE_DIR)/dataset \
	       $(RECBOLE_DIR)/saved $(RECBOLE_DIR)/log \
	       $(RECBOLE_DIR)/log_tensorboard
	@echo "note: reproduced table.tex is kept."

clean-envs:
	-$(MAMBA) env remove -n $(ENV_ELLIOT) -y
	-$(MAMBA) env remove -n $(ENV_RECBOLE) -y
	-$(MAMBA) env remove -n $(ENV_WARPREC) -y
