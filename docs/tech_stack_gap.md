# Tech Stack Gap Map

This repository is a bridge project for the ETH/RSL project described in
`Rock_breaking_student_project.pdf`: **AI-Driven Rock Reshaping Simulation and
Control**.

The PDF project combines FEM fracture simulation, supervised GNN learning, and
RL control to guide rock breaking toward a desired geometry. This repository is
intentionally smaller. It uses a 2D lattice simulator and model-based planning to
exercise the same system shape before moving to the full stack.

## What This Repository Demonstrates

| PDF project need | Current repository coverage | Evidence in this repo |
| --- | --- | --- |
| Python implementation and learning framework experience | Implemented in Python with PyTorch, NumPy, YAML configs, W&B logging | `src/`, `scripts/`, `requirements.txt`, `configs/` |
| Graph representation of fracture state | Converts lattice state, edge state, node survival, and impact action into graph transition records | `src/graph_dataset.py`, `src/lattice_sim.py` |
| GNN-based fracture surrogate | MeshGraphNet-style model predicts edge failures and next node-alive state | `src/models.py`, `scripts/train_surrogate.py` |
| Target-driven rock reshaping | Binary target masks and rollout metrics evaluate final shape quality | `src/metrics.py`, `configs/experiment/` |
| Action selection for impact control | CEM planner searches impact sequences through the learned surrogate | `src/planner.py`, `scripts/eval_rollout.py` |
| Experiment discipline under compute limits | Balanced 24x24 run, AWS notes, W&B histories, reports, and rollout figures | `README.md`, `docs/aws.md`, `results/` |

## What Is Still Missing For The PDF Project

| Missing stack | Why it matters for the PDF project | Suggested next bridge step |
| --- | --- | --- |
| FEM fracture simulation | The PDF project expects finite element stress and crack propagation rather than a toy lattice rule | Add a simulator adapter interface and document how FEM outputs would become graph transitions |
| 3D rock shapes | The PDF work packages mention both 2D and 3D rock shapes | Add a small 3D voxel or tetrahedral-graph prototype before integrating real FEM meshes |
| Reinforcement learning | The PDF explicitly asks for an RL agent selecting impact points or drilling patterns | Wrap the simulator/surrogate as a Gymnasium-style environment and add a small PPO/SAC baseline |
| Online parameter adaptation | The PDF includes online adaptation of rock parameters from observed fracture patterns | Add a material-parameter inference experiment for threshold, radius, and damage parameters |
| JAX ecosystem familiarity | JAX is listed as a plus in the requirements | Either port one surrogate component to JAX/Flax or write a short JAX operator-learning spike |
| Realistic geometry and materials | The PDF goal is generalization across geometries and materials | Add randomized material fields, irregular boundaries, and geometry-conditioned train/test splits |
| Surrogate-vs-simulator mismatch correction | Full control depends on reliable long-horizon surrogate rollouts | Add calibration, uncertainty estimates, or model ensembles for planner risk control |

## Recommended Positioning

Use this repository to say:

> I built a compact preparation project that mirrors the project structure:
> fracture data generation, graph surrogate learning, and target-driven control.
> It demonstrates the GNN/control part of the stack and makes the remaining
> gaps explicit: FEM, 3D geometry, RL, online adaptation, and JAX.

Avoid positioning it as:

> A realistic FEM fracture simulator or a complete replacement for the proposed
> RSL rock reshaping project.

## Best Next Additions

1. Add an RL environment wrapper around the current simulator and surrogate.
2. Add a small online material-parameter adaptation experiment.
3. Add a `SimulatorAdapter` abstraction so lattice and future FEM backends share
   one graph-transition interface.
4. Add a minimal 3D graph/voxel data path to show the extension from 2D to 3D.
5. Add a short JAX spike if the application needs visible JAX familiarity.
