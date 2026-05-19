# Findings And Experiment Notes

## Research Question

Can a graph neural surrogate learn enough local fracture dynamics from a rule-based lattice simulator to support target-driven impact planning?

The benchmark is designed to separate three questions:

1. whether one-step edge failure is learnable,
2. whether predicted node survival stays accurate enough over short rollouts,
3. whether CEM planned through the learned surrogate transfers to the simulator.

## Model

The surrogate is a MeshGraphNet-style message passing network with shared latent graph processing and two task heads:

- `edge_logits`: probability that an intact edge breaks after the impact,
- `node_logits`: probability that each node remains in the connected rock after the impact.

The edge head captures local fracture events. The node head gives the planner a shape-level signal and helps diagnose rollout drift.

## Current Evaluation

Run:

```bash
.venv/bin/python scripts/run_full_experiment.py --preset balanced
```

The current balanced 24x24 setting evaluates four target geometries across multiple seeds. The completed sweep used AWS EC2 instance type `g5.2xlarge`; the exact AMI ID was not recorded in this checkout, but the environment was a PyTorch-capable AWS Deep Learning AMI.

- triangle wedge,
- diagonal bevel,
- rectangle cut,
- circular notch.

Baselines:

- random action sequences,
- greedy one-step simulator planning,
- CEM through the learned surrogate,
- CEM through the simulator oracle.

Outputs:

- `results/rollouts/balanced/eval_metrics.csv`
- `results/rollouts/balanced/eval_summary.csv`
- `results/rollouts/balanced/report.html`

## Reporting Criteria

The result should be considered strong if:

- validation edge F1 is above 0.75,
- validation node-alive IoU is above 0.85,
- five-step surrogate planning beats random on most target/seed pairs,
- oracle CEM remains the upper bound,
- surrogate CEM narrows a visible portion of the gap between greedy and oracle CEM.

## Planned Ablations

- 24x24 versus 32x32 or 48x48 lattices when stronger hardware is available.
- 128 versus 192 hidden dimensions.
- edge-only rollout versus edge-plus-node rollout.
- CEM population and horizon sweeps.
