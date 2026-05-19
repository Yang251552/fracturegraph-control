# FractureGraph-Control

FractureGraph-Control is a small preparation project for the RSL topic
**AI-Driven Rock Reshaping Simulation and Control**. The project description
combines FEM fracture simulation, supervised GNN dynamics learning, and an RL
controller that chooses impact or drilling actions toward a desired rock shape.

This repository implements a smaller 2D version of that pipeline:

```text
2D lattice fracture simulator
        -> graph transition dataset
        -> MeshGraphNet-style dynamics surrogate
        -> CEM planner for target-shape control
```

The goal is not to reproduce FEM fracture mechanics. The goal is to make the
main data and control interfaces concrete: generate fracture transitions, train
a graph surrogate, plan through the learned model, and measure the final shape
against a target geometry.

This is the fourth repo in a small preparation set:

| Repo | Role |
| --- | --- |
| [`isaac-lab-manipulation`](https://github.com/Yang251552/isaac-lab-manipulation) | Standard Isaac Lab + `rsl_rl` manipulation baseline. |
| [`excavation-rl`](https://github.com/Yang251552/excavation-rl) | Granular excavation attempt with a self-built simulation and training loop. |
| [`cluttered-lift`](https://github.com/Yang251552/cluttered-lift) | Contact-rich manipulation diagnostic inside Isaac Lab using a rigid-body granular proxy. |
| this repo | Graph fracture surrogate and target-shape planning loop. |

## Project Fit

| Target project component | This repository |
| --- | --- |
| FEM fracture simulation | Rule-based 2D lattice simulator with hidden edge thresholds, anisotropic stress, damage accumulation, and fragment removal. |
| Supervised GNN training on fracture data | Graph transition records with node, edge, and global action features; MeshGraphNet-style model predicts edge breaks and next alive nodes. |
| RL agent choosing impact or drilling actions | Model-based CEM planner over impact point and force. This is a planning substitute for the RL part, not a full RL implementation. |
| Desired rock geometry | Binary target masks with IoU, undercut, overbreak, and force-cost metrics. |
| 2D/3D shapes and online material adaptation | Current implementation is 2D only. 3D geometry and parameter adaptation are left as next steps. |

Within the set, this repo is the one closest to the final proposal's
fracture-prediction and target-shape-control loop. The earlier repos cover the
robot-learning baseline and contact-rich manipulation context.

## Implemented Components

- Config-driven simulator, dataset generation, training, planning, and
  evaluation scripts.
- 2D lattice fracture simulator with stochastic material thresholds and
  accumulated damage.
- Graph dataset format for fracture transitions:
  - node features: `[x, y, boundary, alive, damage]`
  - edge features: `[dx, dy, length, broken]`
  - global action: `[impact_x, impact_y, force]`
  - labels: newly broken edges and next-step alive nodes
- MeshGraphNet-style surrogate with two heads:
  - edge failure probability
  - next-step node alive probability
- CEM planner that can roll out either the learned surrogate or the simulator
  oracle.
- Multi-target evaluation with random, greedy, surrogate-CEM, and oracle-CEM
  baselines.

## Current Result

The current result package uses the balanced `24x24` setting. Larger `32x32`
and `48x48` configs are present, but the completed run was kept at `24x24`
because it was the practical scale for the available GPU budget.

Selected surrogate run:

- W&B run: [`524mg3ns`](https://wandb.ai/yangchenghan2515-eth-z-rich/fracturegraph-control/runs/524mg3ns)
- validation edge F1: `0.871`
- validation selection score: `0.9157`

The selected checkpoint used for the recorded rollout was:

```text
results/checkpoints/balanced_monitor_20260518_213730/surrogate_best.pt
```

This checkpoint is not included in the repository. The repo keeps generated
figures and rollout reports, but ignores datasets and model checkpoints.

Surrogate-CEM rollout summary:

| Target | Mean final IoU | Main observation |
| --- | ---: | --- |
| `triangle_wedge` | `0.9773` | Transfers well. |
| `diagonal_bevel` | `0.9869` | Transfers well. |
| `rectangle_cut` | `0.9539` | Planner tends to leave some target-removal cells alive. |
| `circular_notch` | `0.9430` | Curved boundary is harder for the current surrogate/planner. |

![Rollout IoU by target](results/figures/balanced_20260519/rollout_iou_by_target.svg)

![Rollout error breakdown](results/figures/balanced_20260519/rollout_error_breakdown.svg)

Representative surrogate rollouts:

| `triangle_wedge.seed_1.cem_surrogate` | `diagonal_bevel.seed_1.cem_surrogate` |
| --- | --- |
| <img src="results/rollouts/balanced_monitor_20260518_213730/triangle_wedge.seed_1.cem_surrogate.svg" width="520"> | <img src="results/rollouts/balanced_monitor_20260518_213730/diagonal_bevel.seed_1.cem_surrogate.svg" width="520"> |

Representative harder cases:

| `rectangle_cut.seed_1.greedy` | `rectangle_cut.seed_1.cem_surrogate` |
| --- | --- |
| <img src="results/rollouts/balanced_monitor_20260518_213730/rectangle_cut.seed_1.greedy.svg" width="520"> | <img src="results/rollouts/balanced_monitor_20260518_213730/rectangle_cut.seed_1.cem_surrogate.svg" width="520"> |

The planner-side threshold sweep is kept in:

- `results/rollouts/planner_sweep_20260519_085313/sweep_summary.csv`
- `results/rollouts/planner_verify_light_20260519_123026/sweep_summary.csv`

The short version is that lowering the surrogate break threshold to `0.90`
helps `rectangle_cut` in some seeds, but was not stable enough to make it a
global default.

## Running The Code

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Development-scale check:

```bash
.venv/bin/python scripts/run_full_experiment.py --preset dev --max-seeds 1
```

Full balanced run:

```bash
.venv/bin/python scripts/run_full_experiment.py --preset balanced
```

This runs dataset generation, surrogate training, and multi-target rollout
evaluation. W&B logging is enabled in `configs/train/surrogate_balanced.yaml`;
use `wandb login` or set `WANDB_MODE=offline` on a fresh machine.

Individual entrypoints:

```bash
.venv/bin/python scripts/generate_dataset.py --config configs/data/lattice_24_balanced.yaml --render-preview
.venv/bin/python scripts/train_surrogate.py \
  --data-config configs/data/lattice_24_balanced.yaml \
  --model-config configs/model/meshgnn_balanced.yaml \
  --train-config configs/train/surrogate_balanced.yaml
.venv/bin/python scripts/eval_rollout.py \
  --experiment configs/experiment/reshape_targets_balanced.yaml \
  --checkpoint results/checkpoints/balanced/surrogate_best.pt \
  --output-dir results/rollouts/balanced
```

## Repository Layout

```text
configs/        Data, model, training, planner, and experiment configs
scripts/        Dataset, training, planning, evaluation, and full-run entrypoints
src/            Simulator, graph data, model, planner, metrics, visualization
docs/           AWS notes, findings notes, and project-fit notes
results/        Generated figures and rollout reports
data/           Generated graph datasets, ignored by Git
```

## Current Limits

- The simulator is a 2D lattice rule, not FEM.
- The controller is CEM planning, not an RL policy.
- The current model does not handle online material-parameter adaptation.
- The stored result package does not include the balanced dataset or selected
  checkpoint.
- The next useful extensions would be a simulator-adapter interface, a small RL
  environment wrapper, material-parameter inference, and a minimal 3D graph or
  voxel version.
