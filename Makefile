.PHONY: dev data train eval plan clean

PYTHON ?= python3

dev:
	$(PYTHON) scripts/run_full_experiment.py --preset dev --max-seeds 1

data:
	$(PYTHON) scripts/generate_dataset.py --config configs/data/lattice_32.yaml --render-preview

train:
	$(PYTHON) scripts/train_surrogate.py

eval:
	$(PYTHON) scripts/eval_rollout.py --experiment configs/experiment/reshape_targets.yaml

plan:
	$(PYTHON) scripts/plan_break.py --method cem_oracle

clean:
	rm -rf data/lattice_16_dev data/lattice_32 results/figures/*.svg results/rollouts/*.svg results/rollouts/*.json results/rollouts/*.html results/rollouts/*.csv
