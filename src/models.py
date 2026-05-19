from __future__ import annotations

from typing import Dict


def require_torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for the learned surrogate. "
            "Install the project dependencies with `python3 -m pip install -r requirements.txt`."
        ) from exc


try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class MLP(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x):
            return self.net(x)


    class FractureGraphNet(nn.Module):
        """MeshGraphNet-style surrogate with edge-failure and node-alive heads."""

        def __init__(
            self,
            node_dim: int = 5,
            edge_dim: int = 4,
            global_dim: int = 3,
            hidden_dim: int = 96,
            message_passing_steps: int = 5,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.message_passing_steps = message_passing_steps
            self.node_encoder = MLP(node_dim, hidden_dim, hidden_dim, dropout)
            self.edge_encoder = MLP(edge_dim + global_dim, hidden_dim, hidden_dim, dropout)
            self.edge_updates = nn.ModuleList(
                [MLP(hidden_dim * 3 + global_dim, hidden_dim, hidden_dim, dropout) for _ in range(message_passing_steps)]
            )
            self.node_updates = nn.ModuleList(
                [MLP(hidden_dim * 2 + global_dim, hidden_dim, hidden_dim, dropout) for _ in range(message_passing_steps)]
            )
            self.edge_decoder = MLP(hidden_dim, hidden_dim, 1, dropout=0.0)
            self.node_decoder = MLP(hidden_dim, hidden_dim, 1, dropout=0.0)

        def forward(self, batch: Dict[str, "torch.Tensor"]):
            x = batch["x"]
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            u = batch["u"]
            node_batch = batch["node_batch"]
            edge_batch = batch["edge_batch"]

            src = edge_index[0]
            dst = edge_index[1]
            node_latent = self.node_encoder(x)
            edge_latent = self.edge_encoder(torch.cat([edge_attr, u[edge_batch]], dim=-1))

            for edge_update, node_update in zip(self.edge_updates, self.node_updates):
                edge_context = torch.cat(
                    [edge_latent, node_latent[src], node_latent[dst], u[edge_batch]],
                    dim=-1,
                )
                edge_delta = edge_update(edge_context)
                edge_latent = edge_latent + edge_delta

                aggregate = torch.zeros_like(node_latent)
                aggregate.index_add_(0, dst, edge_latent)
                aggregate.index_add_(0, src, edge_latent)

                degree = torch.zeros((node_latent.shape[0], 1), dtype=node_latent.dtype, device=node_latent.device)
                ones = torch.ones((edge_latent.shape[0], 1), dtype=node_latent.dtype, device=node_latent.device)
                degree.index_add_(0, dst, ones)
                degree.index_add_(0, src, ones)
                aggregate = aggregate / degree.clamp_min(1.0)

                node_context = torch.cat([node_latent, aggregate, u[node_batch]], dim=-1)
                node_latent = node_latent + node_update(node_context)

            return {
                "edge_logits": self.edge_decoder(edge_latent).squeeze(-1),
                "node_logits": self.node_decoder(node_latent).squeeze(-1),
            }


def build_model(model_config: Dict[str, object]):
    require_torch()
    hidden_dim = int(model_config.get("hidden_dim", 128))
    steps = int(model_config.get("message_passing_steps", 8))
    dropout = float(model_config.get("dropout", 0.0))
    return FractureGraphNet(hidden_dim=hidden_dim, message_passing_steps=steps, dropout=dropout)
