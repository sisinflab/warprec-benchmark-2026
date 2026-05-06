# Makefile — Reproducibility driver for Table 5
# ================================================================
#
# End-to-end reproduction of Table 5 (cross-framework evaluation of
# ItemKNN and LightGCN on MovieLens-1M, cutoff@10, epoch 1) across
# Elliot, RecBole, and WarpRec.
#
#   make help        # list targets
#   make all         # full pipeline
#   make install     # create the 3 conda envs
#   make clean       # remove generated artefacts (keeps envs)
#
# Variables:
#   CONDA_EXE = micromamba | mamba | conda  [default: first found]
# ================================================================

CONDA_EXE ?= $(shell \
	if [ -n "$$MAMBA_EXE" ] && [ -x "$$MAMBA_EXE" ]; then echo "$$MAMBA_EXE"; exit 0; fi; \
	for c in micromamba mamba conda; do \
		command -v $$c >/dev/null 2>&1 && { echo $$c; exit 0; }; \
	done; \
	for p in $$HOME/.local/bin/micromamba $$HOME/bin/micromamba \
	         $$HOME/micromamba/bin/micromamba /opt/micromamba/bin/micromamba \
	         /usr/local/bin/micromamba; do \
		[ -x "$$p" ] && { echo "$$p"; exit 0; }; \
	done)

PY ?= python

ENV_ELLIOT  := table_5_reproducibility_elliot
ENV_RECBOLE := table_5_reproducibility_recbole
ENV_WARPREC := table_5_reproducibility_warprec

ENV_YML_ELLIOT  := environments/elliot.yml
ENV_YML_RECBOLE := environments/recbole.yml
ENV_YML_WARPREC := environments/warprec.yml

RUN_ELLIOT  = $(CONDA_EXE) run -n $(ENV_ELLIOT)
RUN_RECBOLE = $(CONDA_EXE) run -n $(ENV_RECBOLE)
RUN_WARPREC = $(CONDA_EXE) run -n $(ENV_WARPREC)

.PHONY: help all install install-elliot install-recbole install-warprec \
        _check-conda \
        train-warprec train-elliot train-recbole \
        cross-eval cross-eval-elliot cross-eval-recbole cross-eval-warprec \
        aggregate clean clean-envs

help:
	@printf '\nWarpRec . Table 5 reproducibility\n\n'
	@printf '  make all              full pipeline\n'
	@printf '  make install          create the 3 conda envs\n'
	@printf '  make train-warprec    train WarpRec (also generates the canonical split)\n'
	@printf '  make train-elliot     train Elliot (consumes WarpRec split)\n'
	@printf '  make train-recbole    train RecBole (consumes WarpRec split)\n'
	@printf '  make cross-eval       run the 6 cross-framework evaluations\n'
	@printf '  make aggregate        build results/table.md and results/table.tex\n'
	@printf '  make clean            remove generated artefacts (keeps envs)\n'
	@printf '  make clean-envs       remove the 3 conda envs\n\n'
	@printf 'CONDA_EXE = $(CONDA_EXE)\n\n'

# ---------------------------------------------------------------
# Env creation
# ---------------------------------------------------------------
install: install-warprec install-recbole install-elliot

_check-conda:
	@if [ -z "$(CONDA_EXE)" ]; then \
		echo "error: no env manager (looked for micromamba, mamba, conda)" >&2; \
		exit 1; \
	fi

# Elliot installs in 2 phases because torch_scatter / torch_sparse are
# source-built and need `import torch` to succeed at build time.
install-elliot: _check-conda
	@echo ">>> [elliot 1/2] base env from $(ENV_YML_ELLIOT)"
	@$(CONDA_EXE) env list | grep -q " $(ENV_ELLIOT) " \
		&& $(CONDA_EXE) env update -n $(ENV_ELLIOT) -f $(ENV_YML_ELLIOT) \
		|| $(CONDA_EXE) env create -n $(ENV_ELLIOT) -f $(ENV_YML_ELLIOT)
	@echo ">>> [elliot 2/2] PyG ecosystem (built against the env's torch)"
	$(RUN_ELLIOT) pip install --no-build-isolation \
		torch_scatter==2.1.2 \
		torch_sparse==0.6.18 \
		torch-geometric==2.6.1 || true

install-recbole: _check-conda
	@echo ">>> creating/updating $(ENV_RECBOLE)"
	@$(CONDA_EXE) env list | grep -q " $(ENV_RECBOLE) " \
		&& $(CONDA_EXE) env update -n $(ENV_RECBOLE) -f $(ENV_YML_RECBOLE) \
		|| $(CONDA_EXE) env create -n $(ENV_RECBOLE) -f $(ENV_YML_RECBOLE)

install-warprec: _check-conda
	@echo ">>> creating/updating $(ENV_WARPREC)"
	@$(CONDA_EXE) env list | grep -q " $(ENV_WARPREC) " \
		&& $(CONDA_EXE) env update -n $(ENV_WARPREC) -f $(ENV_YML_WARPREC) \
		|| $(CONDA_EXE) env create -n $(ENV_WARPREC) -f $(ENV_YML_WARPREC)

# ---------------------------------------------------------------
# Stage 1: WarpRec training + canonical split export
# ---------------------------------------------------------------
train-warprec:
	@echo ">>> [warprec] training ItemKNN + LightGCN, exporting canonical split"
	$(RUN_WARPREC) $(PY) -m warprec.run -c frameworks/warprec/train.yml -p train
	$(RUN_WARPREC) $(PY) scripts/export_split.py
	$(RUN_WARPREC) $(PY) scripts/collect_recs.py warprec

# ---------------------------------------------------------------
# Stage 2: Elliot training (consumes data/split/*.tsv)
# ---------------------------------------------------------------
train-elliot:
	@echo ">>> [elliot] training ItemKNN + LightGCN"
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py frameworks/elliot/train.yml
	$(RUN_ELLIOT) $(PY) scripts/collect_recs.py elliot

# ---------------------------------------------------------------
# Stage 3: RecBole training (consumes data/split/*.tsv)
# ---------------------------------------------------------------
train-recbole:
	@echo ">>> [recbole] staging splits and training"
	$(RUN_RECBOLE) $(PY) frameworks/recbole/run_experiment.py
	$(RUN_RECBOLE) $(PY) scripts/collect_recs.py recbole

# ---------------------------------------------------------------
# Stage 4: Cross-framework evaluation
# ---------------------------------------------------------------
cross-eval: cross-eval-elliot cross-eval-recbole cross-eval-warprec

cross-eval-elliot:
	@echo ">>> [elliot] cross-evaluating RecBole + WarpRec recommendations"
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py frameworks/elliot/eval_recbole_itemknn.yml
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py frameworks/elliot/eval_recbole_lightgcn.yml
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py frameworks/elliot/eval_warprec_itemknn.yml
	$(RUN_ELLIOT) $(PY) scripts/run_elliot.py frameworks/elliot/eval_warprec_lightgcn.yml
	$(RUN_ELLIOT) $(PY) scripts/collect_metrics.py elliot  elliot
	$(RUN_ELLIOT) $(PY) scripts/collect_metrics.py recbole elliot
	$(RUN_ELLIOT) $(PY) scripts/collect_metrics.py warprec elliot

cross-eval-recbole:
	@echo ">>> [recbole] cross-evaluating Elliot + WarpRec recommendations"
	$(RUN_RECBOLE) $(PY) frameworks/recbole/evaluate_external_recs.py
	$(RUN_RECBOLE) $(PY) scripts/collect_metrics.py recbole recbole
	$(RUN_RECBOLE) $(PY) scripts/collect_metrics.py elliot  recbole
	$(RUN_RECBOLE) $(PY) scripts/collect_metrics.py warprec recbole

cross-eval-warprec:
	@echo ">>> [warprec] cross-evaluating Elliot + RecBole recommendations"
	$(RUN_WARPREC) $(PY) -m warprec.run -c frameworks/warprec/eval_elliot_itemknn.yml  -p eval
	$(RUN_WARPREC) $(PY) -m warprec.run -c frameworks/warprec/eval_elliot_lightgcn.yml -p eval
	$(RUN_WARPREC) $(PY) -m warprec.run -c frameworks/warprec/eval_recbole_itemknn.yml  -p eval
	$(RUN_WARPREC) $(PY) -m warprec.run -c frameworks/warprec/eval_recbole_lightgcn.yml -p eval
	$(RUN_WARPREC) $(PY) scripts/collect_metrics.py warprec warprec
	$(RUN_WARPREC) $(PY) scripts/collect_metrics.py elliot  warprec
	$(RUN_WARPREC) $(PY) scripts/collect_metrics.py recbole warprec

# ---------------------------------------------------------------
# Stage 5: Aggregate
# ---------------------------------------------------------------
aggregate:
	$(PY) scripts/aggregate_results.py

# ---------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------
all: train-warprec train-elliot train-recbole cross-eval aggregate
	@echo ">>> done. See results/table.md and results/table.tex."

# ---------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------
clean:
	rm -rf data/split/*.tsv
	rm -rf results/experiment results/elliot_runs results/recbole_runs \
	       results/recs results/metrics results/table.md results/table.tex
	@echo ">>> cleaned generated artefacts (envs and raw data preserved)"

clean-envs:
	-$(CONDA_EXE) env remove -n $(ENV_ELLIOT) -y
	-$(CONDA_EXE) env remove -n $(ENV_RECBOLE) -y
	-$(CONDA_EXE) env remove -n $(ENV_WARPREC) -y
