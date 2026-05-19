# Cloud Training Notes

The current deliverable baseline is the balanced 24x24 experiment. It was chosen because larger 32x32 runs were not reliable on the available hardware. The completed balanced sweep used AWS EC2 instance type `g5.2xlarge`; the exact AMI ID was not recorded in this checkout, but the environment was a PyTorch-capable AWS Deep Learning AMI. A single GPU cloud instance is useful if the goal is to reproduce the 24x24 run faster or scale the benchmark back up to 32x32 or 48x48.

## Recommended Instance

- Use EC2 instance type `g5.2xlarge` to match the completed 24x24 sweep environment.
- AWS EC2 API reports `g5.2xlarge` as 8 vCPUs, 32 GiB memory, and one NVIDIA A10G GPU with about 22.9 GiB GPU memory.
- Use a PyTorch-capable AWS Deep Learning AMI with NVIDIA GPU drivers, and record the exact AMI ID for future runs.
- Smaller single-GPU instances may be enough for smoke checks, but they are not the documented baseline.
- 100-200 GB EBS volume if keeping generated datasets and checkpoints.

## Setup

```bash
git clone <repo-url> fracturegraph-control
cd fracturegraph-control
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The balanced training config enables W&B logging. Run `wandb login` before training, or set `WANDB_MODE=offline` if online logging is not needed.

## Balanced 24x24 Run

```bash
.venv/bin/python scripts/run_full_experiment.py --preset balanced
```

Main outputs:

- `data/lattice_24_balanced/`
- `results/checkpoints/balanced/surrogate_best.pt`
- `results/rollouts/balanced/eval_metrics.csv`
- `results/rollouts/balanced/eval_summary.csv`
- `results/rollouts/balanced/report.html`

## Faster Verification Run

```bash
.venv/bin/python scripts/run_full_experiment.py --preset dev --max-seeds 1
```

Use this only to confirm the environment. The balanced 24x24 run is the current project-reporting baseline.

## Scaling Runs

The repository still includes 32x32 and 48x48 presets. Treat them as follow-up experiments for a stronger GPU environment, not as the current deliverable baseline.
