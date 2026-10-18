"""
model.py - Multi-Task Heavy Ranking Network (Saves / Dwell Time / Private Shares)

One file, three tasks:
    Task 1  Residual DCN block - low-rank CrossNet (W ~= U @ V^T) stacked with residuals
    Task 2  Three task heads (Saves, Dwell Time, Private Shares) + weighted joint loss
    Task 3  MLflow tracking (joint loss + per-task AUC every epoch) and programmatic
            registration of the trained model in the Azure ML model registry

Where it runs
    * Databricks notebook -> MLflow experiment at /Users/<you>/multi_task_ranking_practice
    * Azure ML            -> MLflow tracking pointed at the workspace (needs azureml-mlflow),
                             or the MLFLOW_TRACKING_URI an Azure ML job already sets
    * Plain Python        -> local MLflow store
Azure ML registration runs whenever an MLClient can be built: from the AZURE_* environment
variables, the constants below, or a config.json found by MLClient.from_config().

Usage
    python model.py                    # sanity checks, train, log to MLflow, register in Azure ML
    python model.py --epochs 20        # override any TrainConfig field from the command line
    python model.py --skip-register    # train and log only

Requirements
    torch, numpy, scikit-learn, matplotlib, mlflow, azure-ai-ml, azure-identity
    (+ azureml-mlflow when tracking straight into an Azure ML workspace)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

try:  # Azure ML SDK v2 - optional, so the script still trains and logs without it
    from azure.ai.ml import MLClient
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from azure.core.exceptions import ClientAuthenticationError
    from azure.identity import DefaultAzureCredential

    AZURE_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    AZURE_SDK_AVAILABLE = False

    class ClientAuthenticationError(Exception):  # placeholder so except clauses stay valid
        pass


# ---------------------------------------------------------------------------
# Azure ML workspace + experiment configuration
# ---------------------------------------------------------------------------
# Paste your values here, or export AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP /
# AZUREML_WORKSPACE_NAME, or drop the workspace config.json next to this file.
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "<your-subscription-id>")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "<your-resource-group>")
WORKSPACE_NAME = os.getenv("AZUREML_WORKSPACE_NAME", "<your-workspace-name>")

EXPERIMENT_NAME = "multi_task_ranking_practice"
REGISTERED_MODEL_NAME = "multi-task-ranking-model"
SUMMARY_FILE = "run_summary.json"


@dataclass
class TrainConfig:
    # Architecture (matches the Task 2 practice block)
    in_features: int = 128
    rank: int = 32
    num_dcn_layers: int = 3
    head_hidden_size: int = 64
    # Optimisation
    epochs: int = 20                  # upper bound; early stopping usually ends sooner
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2        # AdamW (decoupled) weight decay
    grad_clip_norm: float = 1.0
    lr_plateau_factor: float = 0.5    # halve the LR when validation joint loss stalls
    lr_plateau_patience: int = 1
    early_stopping_patience: int = 3  # epochs without a combined-AUC gain before stopping
    # Joint-loss weights: MSE sits on a larger scale than BCE, so Dwell Time is down-weighted
    saves_loss_weight: float = 1.0
    dwell_loss_weight: float = 0.1
    shares_loss_weight: float = 1.0
    # Data
    n_samples: int = 200_000
    val_fraction: float = 0.2
    dwell_auc_quantile: float = 0.5  # "long dwell" = at or above the training median
    # Negative-transfer monitor: auc_saves up while auc_shares down for N consecutive epochs
    neg_transfer_patience: int = 2
    neg_transfer_tolerance: float = 1e-3
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Task 1 - Residual DCN block
# ===========================================================================
class CrossNet(nn.Module):
    """
    A single cross-layer for the DCN.
    Equation: x_{l+1} = x_0 * (x_l^T @ W) + b + x_l
    Low-rank approximation: W = U @ V^T
    """

    def __init__(self, in_features: int, rank: int):
        super(CrossNet, self).__init__()
        self.rank = rank
        # Low-rank projection matrices
        self.U = nn.Parameter(torch.randn(in_features, rank))
        self.V = nn.Parameter(torch.randn(in_features, rank))
        self.bias = nn.Parameter(torch.randn(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Unit-variance randn makes x_l @ V @ U.T roughly in_features * rank (~4,096x) larger
        # than x_l, and the blow-up compounds across stacked layers. Xavier keeps each cross
        # term near the scale of its input so the residual stack trains from step one.
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.bias)

    def forward(self, x_0: torch.Tensor, x_l: torch.Tensor) -> torch.Tensor:
        """
        x_0: The original input tensor to the DCN block.   (batch, in_features)
        x_l: The output tensor from the previous layer.    (batch, in_features)
        """
        # --- Your code here ---
        # 1. Project x_l down to the rank dimension using V.    (B, d) @ (d, r) -> (B, r)
        low_rank = torch.matmul(x_l, self.V)
        # 2. Project the result back up to in_features using U. (B, r) @ (r, d) -> (B, d)
        projection_output = torch.matmul(low_rank, self.U.T)  # == x_l @ V @ U.T
        # 3. Multiply element-wise with x_0.
        # 4. Add the bias and the residual connection (x_l).
        output = x_0 * projection_output + self.bias + x_l
        # --- End of your code ---
        return output


class ResidualDCN(nn.Module):
    def __init__(self, in_features: int, rank: int, num_layers: int):
        super(ResidualDCN, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([CrossNet(in_features, rank) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = x
        x_l = x
        for i in range(self.num_layers):
            x_l = self.cross_layers[i](x_0, x_l)
        return x_l


# ===========================================================================
# Task 2 - Multi-task heads and joint loss
# ===========================================================================
def _make_head(in_features: int, hidden: int) -> nn.Sequential:
    """Linear -> ReLU -> Linear(1): one small tower per engagement objective."""
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class MultiTaskRankingModel(nn.Module):
    """
    Shared Residual DCN trunk feeding three task-specific heads.
    Outputs: (saves_logit, dwell_pred, shares_logit). Saves and Shares are raw logits
    (apply sigmoid for probabilities); Dwell Time is the standardised log-dwell prediction.
    """

    def __init__(self, in_features: int, rank: int, num_dcn_layers: int, head_hidden_size: int):
        super(MultiTaskRankingModel, self).__init__()
        self.dcn_block = ResidualDCN(in_features, rank, num_dcn_layers)
        # --- Your code here ---
        # 1./2. Saves and Shares: 1 output for binary classification.
        # 3.    Dwell Time: 1 output for regression.
        self.saves_head = _make_head(in_features, head_hidden_size)
        self.dwell_time_head = _make_head(in_features, head_hidden_size)
        self.shares_head = _make_head(in_features, head_hidden_size)
        # --- End of your code ---

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dcn_output = self.dcn_block(x)
        # Pass the DCN output to each head
        saves_pred = self.saves_head(dcn_output)
        dwell_pred = self.dwell_time_head(dcn_output)
        shares_pred = self.shares_head(dcn_output)
        return saves_pred, dwell_pred, shares_pred


def compute_joint_loss(
    preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    loss_bce: nn.Module,
    loss_mse: nn.Module,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """joint_loss = w_saves * BCE(saves) + w_dwell * MSE(dwell) + w_shares * BCE(shares)"""
    saves_pred, dwell_pred, shares_pred = preds
    saves_true, dwell_true, shares_true = targets
    loss_saves = loss_bce(saves_pred, saves_true)
    loss_dwell = loss_mse(dwell_pred, dwell_true)
    loss_shares = loss_bce(shares_pred, shares_true)
    joint_loss = (
        cfg.saves_loss_weight * loss_saves
        + cfg.dwell_loss_weight * loss_dwell  # 0.1 by default
        + cfg.shares_loss_weight * loss_shares
    )
    return joint_loss, {"loss_saves": loss_saves, "loss_dwell": loss_dwell, "loss_shares": loss_shares}


# ===========================================================================
# Sanity checks (the "Run the Test" steps from Tasks 1 and 2)
# ===========================================================================
def run_sanity_checks(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    # Task 1: the Residual DCN must not change the shape of its input.
    block = ResidualDCN(cfg.in_features, cfg.rank, cfg.num_dcn_layers)
    x = torch.randn(64, cfg.in_features)
    out = block(x)
    assert out.shape == x.shape, f"ResidualDCN changed shape: {tuple(x.shape)} -> {tuple(out.shape)}"
    print(f"[check] ResidualDCN preserves shape: {tuple(x.shape)} -> {tuple(out.shape)}")

    # Task 2: three heads, one output each, and a finite positive joint loss on dummy labels.
    dummy_model = MultiTaskRankingModel(cfg.in_features, cfg.rank, cfg.num_dcn_layers, cfg.head_hidden_size)
    dummy_input = torch.randn(64, cfg.in_features)
    saves_true = torch.randint(0, 2, (64, 1)).float()
    dwell_true = torch.randn(64, 1)
    shares_true = torch.randint(0, 2, (64, 1)).float()
    preds = dummy_model(dummy_input)
    assert all(p.shape == (64, 1) for p in preds), [tuple(p.shape) for p in preds]
    joint_loss, _ = compute_joint_loss(
        preds, (saves_true, dwell_true, shares_true), nn.BCEWithLogitsLoss(), nn.MSELoss(), cfg
    )
    assert torch.isfinite(joint_loss) and joint_loss.item() > 0
    print(f"[check] Joint loss calculated: {joint_loss.item() > 0} ({joint_loss.item():.4f})")


# ===========================================================================
# Data - synthetic user x item engagement logs
# ===========================================================================
def make_synthetic_engagement_data(n_samples: int, n_features: int, seed: int) -> Dict[str, np.ndarray]:
    """
    Stand-in for joined user/item feature vectors with three engagement labels.
    Labels depend on pairwise feature crosses (the signal a DCN is built to learn) and are
    only partly aligned: the "flashy" cross raises Saves but lowers Private Shares, so a
    naive shared model can trade the rare Shares task away for the frequent Saves task.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    def group_score(lo: int, hi: int) -> np.ndarray:
        w = rng.normal(0.0, 1.0, hi - lo) / np.sqrt(hi - lo)
        return X[:, lo:hi] @ w

    quality = group_score(0, 16)        # general item quality / user affinity
    depth = group_score(16, 32)         # long-form, topical depth signals
    personal = group_score(32, 48)      # closeness of item to the user's own network
    cross_flashy = X[:, 48] * X[:, 49]  # flashy creative x click-happy user
    cross_depth = X[:, 50] * X[:, 51]   # topic depth x user expertise
    cross_personal = X[:, 52] * X[:, 53]

    saves_logit = 1.4 * quality + 1.0 * cross_flashy + 0.3 * cross_depth - 1.1
    log_dwell = (
        3.0 + 0.5 * quality + 0.7 * depth + 0.8 * cross_depth + 0.2 * cross_flashy
        + rng.normal(0.0, 0.6, n_samples)
    )
    shares_logit = 1.2 * personal + 1.3 * cross_personal + 0.5 * quality - 0.8 * cross_flashy - 4.1

    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    saves = (rng.random(n_samples) < sigmoid(saves_logit)).astype(np.float32)
    shares = (rng.random(n_samples) < sigmoid(shares_logit)).astype(np.float32)
    dwell_seconds = np.exp(log_dwell).astype(np.float32)
    return {"X": X, "saves": saves, "dwell_seconds": dwell_seconds, "shares": shares}


def build_loaders(cfg: TrainConfig):
    data = make_synthetic_engagement_data(cfg.n_samples, cfg.in_features, cfg.seed)
    n_val = int(cfg.n_samples * cfg.val_fraction)
    perm = np.random.default_rng(cfg.seed + 1).permutation(cfg.n_samples)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Dwell target: log seconds, standardised with TRAIN statistics only (no leakage).
    log_dwell = np.log(data["dwell_seconds"])
    mu, sigma = log_dwell[train_idx].mean(), log_dwell[train_idx].std()
    dwell_z = ((log_dwell - mu) / sigma).astype(np.float32)
    long_dwell_threshold = float(np.quantile(dwell_z[train_idx], cfg.dwell_auc_quantile))

    def to_ds(idx):
        return TensorDataset(
            torch.from_numpy(data["X"][idx]),
            torch.from_numpy(data["saves"][idx]).unsqueeze(1),
            torch.from_numpy(dwell_z[idx]).unsqueeze(1),
            torch.from_numpy(data["shares"][idx]).unsqueeze(1),
        )

    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(to_ds(train_idx), batch_size=cfg.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(to_ds(val_idx), batch_size=4096, shuffle=False)
    stats = {
        "n_train": len(train_idx),
        "n_val": n_val,
        "saves_rate": float(data["saves"][train_idx].mean()),
        "shares_rate": float(data["shares"][train_idx].mean()),
        "dwell_median_seconds": float(np.median(data["dwell_seconds"][train_idx])),
        "dwell_log_mean": float(mu),
        "dwell_log_std": float(sigma),
        "long_dwell_threshold_z": long_dwell_threshold,
    }
    return train_loader, val_loader, stats


# ===========================================================================
# Training and evaluation
# ===========================================================================
def train_one_epoch(model, loader, optimizer, loss_bce, loss_mse, cfg, device) -> Dict[str, float]:
    model.train()
    totals = {"joint_loss": 0.0, "loss_saves": 0.0, "loss_dwell": 0.0, "loss_shares": 0.0}
    n_seen = 0
    for x, y_saves, y_dwell, y_shares in loader:
        x, y_saves, y_dwell, y_shares = (t.to(device) for t in (x, y_saves, y_dwell, y_shares))
        optimizer.zero_grad()
        preds = model(x)
        joint_loss, parts = compute_joint_loss(preds, (y_saves, y_dwell, y_shares), loss_bce, loss_mse, cfg)
        joint_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        bs = x.size(0)
        n_seen += bs
        totals["joint_loss"] += joint_loss.item() * bs
        for k, v in parts.items():
            totals[k] += v.item() * bs
    return {k: v / n_seen for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, loss_bce, loss_mse, cfg, long_dwell_threshold, device) -> Dict[str, float]:
    model.eval()
    totals = {"joint_loss": 0.0, "loss_saves": 0.0, "loss_dwell": 0.0, "loss_shares": 0.0}
    cols = {k: [] for k in ("saves_p", "saves_y", "dwell_p", "dwell_y", "shares_p", "shares_y")}
    n_seen = 0
    for x, y_saves, y_dwell, y_shares in loader:
        x, y_saves, y_dwell, y_shares = (t.to(device) for t in (x, y_saves, y_dwell, y_shares))
        preds = model(x)
        joint_loss, parts = compute_joint_loss(preds, (y_saves, y_dwell, y_shares), loss_bce, loss_mse, cfg)
        bs = x.size(0)
        n_seen += bs
        totals["joint_loss"] += joint_loss.item() * bs
        for k, v in parts.items():
            totals[k] += v.item() * bs
        cols["saves_p"].append(torch.sigmoid(preds[0]).cpu())
        cols["dwell_p"].append(preds[1].cpu())
        cols["shares_p"].append(torch.sigmoid(preds[2]).cpu())
        cols["saves_y"].append(y_saves.cpu())
        cols["dwell_y"].append(y_dwell.cpu())
        cols["shares_y"].append(y_shares.cpu())

    c = {k: torch.cat(v).numpy().ravel() for k, v in cols.items()}
    metrics = {f"val_{k}": v / n_seen for k, v in totals.items()}
    metrics["auc_saves"] = float(roc_auc_score(c["saves_y"], c["saves_p"]))
    metrics["auc_shares"] = float(roc_auc_score(c["shares_y"], c["shares_p"]))
    # Dwell Time is a regression head; its AUC ranks items by predicted dwell against the
    # binary "long dwell" label (at or above the training median).
    metrics["auc_dwell"] = float(roc_auc_score((c["dwell_y"] >= long_dwell_threshold).astype(int), c["dwell_p"]))
    metrics["combined_auc"] = float(np.mean([metrics["auc_saves"], metrics["auc_dwell"], metrics["auc_shares"]]))
    metrics["rmse_dwell"] = float(np.sqrt(np.mean((c["dwell_p"] - c["dwell_y"]) ** 2)))
    return metrics


class NegativeTransferMonitor:
    """Flags epochs where Saves keeps improving while Private Shares keeps degrading."""

    def __init__(self, patience: int, tolerance: float):
        self.patience, self.tolerance = patience, tolerance
        self.prev: Optional[Dict[str, float]] = None
        self.streak = 0
        self.ever_flagged = False

    def update(self, m: Dict[str, float]) -> int:
        flag = 0
        if self.prev is not None:
            saves_up = m["auc_saves"] - self.prev["auc_saves"] > self.tolerance
            shares_down = self.prev["auc_shares"] - m["auc_shares"] > self.tolerance
            self.streak = self.streak + 1 if (saves_up and shares_down) else 0
            flag = int(self.streak >= self.patience)
            self.ever_flagged |= bool(flag)
        self.prev = {"auc_saves": m["auc_saves"], "auc_shares": m["auc_shares"]}
        return flag


def plot_training_curves(history, path: str, best_epoch: Optional[int] = None) -> None:
    """Per-task validation AUC and joint loss by epoch, side by side (one y-scale per panel)."""
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = [("auc_saves", "Saves", "#2a78d6"), ("auc_dwell", "Dwell Time", "#eb6834"),
              ("auc_shares", "Private Shares", "#1baf7a")]
    epochs = [h["epoch"] for h in history]
    plt.rcParams["font.family"] = "DejaVu Sans"  # bundled with matplotlib, so identical everywhere
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200, facecolor=surface)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.set_xlabel("Epoch", color=ink2, fontsize=9)
        ax.set_xticks(epochs)
        if best_epoch is not None:
            ax.axvline(best_epoch, color=ink2, linewidth=1, linestyle=(0, (3, 3)), zorder=0)
    for key, label, color in series:
        ys = [h[key] for h in history]
        ax1.plot(epochs, ys, color=color, linewidth=2, marker="o", markersize=4, label=label)
        ax1.annotate(label, (epochs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=8, color=ink)
    ax1.set_xlim(epochs[0] - 0.4, epochs[-1] + 1.6)
    ax1.set_title("Validation AUC by task", loc="left", color=ink, fontsize=10, pad=22)
    ax1.legend(frameon=False, fontsize=8, labelcolor=ink, ncol=3, loc="lower left",
               bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.6)
    for key, label, color in (("train_joint_loss", "Train", "#2a78d6"), ("val_joint_loss", "Validation", "#eb6834")):
        ax2.plot(epochs, [h[key] for h in history], color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax2.set_xlim(epochs[0] - 0.4, epochs[-1] + 0.4)
    ax2.set_title("Joint loss", loc="left", color=ink, fontsize=10, pad=22)
    ax2.legend(frameon=False, fontsize=8, labelcolor=ink, ncol=2, loc="lower left",
               bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.6)
    if best_epoch is not None:
        ax2.annotate(f"best checkpoint (epoch {best_epoch})", xy=(best_epoch, 1), xycoords=("data", "axes fraction"),
                     xytext=(4, -10), textcoords="offset points", fontsize=7.5, color=ink2, va="top")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# ===========================================================================
# Task 3 - MLflow tracking + Azure ML registration
# ===========================================================================
def _placeholder(value: str) -> bool:
    return (not value) or value.startswith("<")


def get_ml_client() -> Optional["MLClient"]:
    """Instantiate the MLClient for the Azure ML workspace, or return None if not configured."""
    if not AZURE_SDK_AVAILABLE:
        print("[azure] azure-ai-ml / azure-identity not installed - registration skipped.")
        return None
    credential = DefaultAzureCredential()
    if not any(_placeholder(v) for v in (SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)):
        return MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
    try:
        return MLClient.from_config(credential=credential)  # looks for config.json
    except Exception:
        print("[azure] No subscription/resource group/workspace configured and no config.json "
              "found - registration skipped.")
        return None


def configure_mlflow(ml_client: Optional["MLClient"]) -> str:
    """Point MLflow at the right tracking store and experiment. Returns an environment label."""
    # Databricks requires an absolute workspace path, e.g. /Users/you@company.com/experiment-name.
    try:
        notebook_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
        current_user = notebook_ctx.userName().get()
        experiment_name = f"/Users/{current_user}/{EXPERIMENT_NAME}"
        mlflow.set_experiment(experiment_name)
        return "databricks"
    except NameError:
        pass  # not running on Databricks

    env = "local"
    if os.getenv("MLFLOW_TRACKING_URI"):
        env = "preset-tracking-uri"  # e.g. inside an Azure ML job, which sets this for you
    elif ml_client is not None:
        try:
            import azureml.mlflow  # noqa: F401  (registers the azureml:// tracking scheme)

            ws = ml_client.workspaces.get(ml_client.workspace_name)
            mlflow.set_tracking_uri(ws.mlflow_tracking_uri)
            env = "azureml"
        except ImportError:
            print("[mlflow] azureml-mlflow not installed - tracking locally; the model will be "
                  "uploaded to Azure ML from the local run artifacts.")
        except ClientAuthenticationError:
            print("[mlflow] Azure authentication failed (run `az login`) - tracking locally.")
        except Exception as exc:  # workspace lookup failed; keep training
            print(f"[mlflow] Could not reach the Azure ML workspace ({exc}) - tracking locally.")
    mlflow.set_experiment(EXPERIMENT_NAME)
    return env


def log_pytorch_model(model: nn.Module, example: torch.Tensor):
    """mlflow.pytorch.log_model with keyword names that work across MLflow 2.x and 3.x."""
    import inspect

    model.eval()
    with torch.no_grad():
        s, d, sh = model(example)
    # Explicit signature: input features -> stacked [saves_logit, dwell_pred, shares_logit].
    signature = infer_signature(example.numpy(), torch.cat([s, d, sh], dim=1).numpy())
    params = inspect.signature(mlflow.pytorch.log_model).parameters
    kwargs = {"signature": signature}
    kwargs["name" if "name" in params else "artifact_path"] = "model"
    if "serialization_format" in params:  # avoids the version-specific input_example error
        kwargs["serialization_format"] = "pickle"
    return mlflow.pytorch.log_model(model.cpu(), **kwargs)


def register_model_in_azure(ml_client, run_id: str, model_uri_fallback: str, metrics: Dict[str, float]):
    """Register the logged MLflow model in the Azure ML registry; returns the registered Model."""
    model_path = f"runs:/{run_id}/model"
    azure_model = Model(
        path=model_path,
        name=REGISTERED_MODEL_NAME,
        type=AssetTypes.MLFLOW_MODEL,
        description="Multi-task heavy ranker: Residual DCN trunk + Saves / Dwell Time / Private Shares heads.",
        tags={
            "mlflow_run_id": run_id,
            "mlflow_model_uri": model_path,
            "auc_saves": f"{metrics['auc_saves']:.4f}",
            "auc_dwell": f"{metrics['auc_dwell']:.4f}",
            "auc_shares": f"{metrics['auc_shares']:.4f}",
            "combined_auc": f"{metrics['combined_auc']:.4f}",
            "framework": "pytorch",
        },
        properties={"mlflow_run_id": run_id, "mlflow_model_uri": model_path},
    )
    try:
        return ml_client.models.create_or_update(azure_model)
    except ClientAuthenticationError:
        raise
    except Exception as exc:
        # runs:/ resolves server-side only when the run lives in this workspace's MLflow store.
        # For Databricks or local tracking, download the model folder and upload it directly;
        # the run ID stays attached through the tags above.
        print(f"[azure] runs:/ registration not resolvable here ({type(exc).__name__}); "
              "uploading the MLflow model folder from the run instead.")
        local_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri_fallback, dst_path=tempfile.mkdtemp(prefix="mtr_model_")
        )
        azure_model.path = local_dir
        return ml_client.models.create_or_update(azure_model)


# ===========================================================================
# Main
# ===========================================================================
def parse_config() -> Tuple[TrainConfig, bool]:
    parser = argparse.ArgumentParser(description="Multi-task heavy ranking network")
    for f in fields(TrainConfig):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    parser.add_argument("--skip-register", action="store_true", help="Train and log without Azure ML registration")
    args, _unknown = parser.parse_known_args()  # tolerate notebook kernel arguments
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    return cfg, args.skip_register


def main() -> Dict:
    cfg, skip_register = parse_config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_sanity_checks(cfg)
    set_seed(cfg.seed)

    # --- Configuration: MLClient + MLflow experiment ---
    ml_client = None if skip_register else get_ml_client()
    tracking_env = configure_mlflow(ml_client)
    print(f"[mlflow] tracking: {tracking_env} -> {mlflow.get_tracking_uri()}")

    train_loader, val_loader, data_stats = build_loaders(cfg)
    print(f"[data] train={data_stats['n_train']:,} val={data_stats['n_val']:,} "
          f"saves={data_stats['saves_rate']:.1%} shares={data_stats['shares_rate']:.1%} "
          f"median dwell={data_stats['dwell_median_seconds']:.1f}s")

    model = MultiTaskRankingModel(cfg.in_features, cfg.rank, cfg.num_dcn_layers, cfg.head_hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_plateau_factor, patience=cfg.lr_plateau_patience)
    loss_bce = nn.BCEWithLogitsLoss()  # For classification (Saves, Shares)
    loss_mse = nn.MSELoss()            # For regression (Dwell Time)
    monitor = NegativeTransferMonitor(cfg.neg_transfer_patience, cfg.neg_transfer_tolerance)

    history = []
    best = {"combined_auc": -1.0, "epoch": 0, "state": None}
    with mlflow.start_run(run_name="multi_task_dcn") as run:
        run_id = run.info.run_id
        mlflow.log_params(asdict(cfg))
        mlflow.log_params({f"data_{k}": v for k, v in data_stats.items()})
        mlflow.set_tags({"task": "multi-task-ranking", "objectives": "saves,dwell_time,private_shares",
                         "architecture": "residual-low-rank-dcn + 3 heads", "tracking_env": tracking_env})
        n_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("n_parameters", n_params)

        # --- Training loop ---
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_m = train_one_epoch(model, train_loader, optimizer, loss_bce, loss_mse, cfg, device)
            val_m = evaluate(model, val_loader, loss_bce, loss_mse, cfg,
                             data_stats["long_dwell_threshold_z"], device)
            neg_flag = monitor.update(val_m)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_m["val_joint_loss"])

            # --- Logging: joint loss + per-task AUC at the end of each epoch ---
            mlflow.log_metric("joint_loss", train_m["joint_loss"], step=epoch)
            mlflow.log_metric("val_joint_loss", val_m["val_joint_loss"], step=epoch)
            mlflow.log_metric("auc_saves", val_m["auc_saves"], step=epoch)
            mlflow.log_metric("auc_dwell", val_m["auc_dwell"], step=epoch)
            mlflow.log_metric("auc_shares", val_m["auc_shares"], step=epoch)
            mlflow.log_metric("combined_auc", val_m["combined_auc"], step=epoch)
            mlflow.log_metrics({
                "train_loss_saves": train_m["loss_saves"],
                "train_loss_dwell": train_m["loss_dwell"],
                "train_loss_shares": train_m["loss_shares"],
                "val_loss_saves": val_m["val_loss_saves"],
                "val_loss_dwell": val_m["val_loss_dwell"],
                "val_loss_shares": val_m["val_loss_shares"],
                "rmse_dwell": val_m["rmse_dwell"],
                "negative_transfer_flag": neg_flag,
                "learning_rate": current_lr,
            }, step=epoch)

            history.append({"epoch": epoch, "train_joint_loss": train_m["joint_loss"], **val_m})
            print(f"epoch {epoch:02d} | joint {train_m['joint_loss']:.4f} (val {val_m['val_joint_loss']:.4f}) | "
                  f"AUC saves {val_m['auc_saves']:.4f} dwell {val_m['auc_dwell']:.4f} "
                  f"shares {val_m['auc_shares']:.4f} | combined {val_m['combined_auc']:.4f} | "
                  f"{time.time() - t0:.1f}s" + ("  <- NEGATIVE TRANSFER" if neg_flag else ""))

            if val_m["combined_auc"] > best["combined_auc"]:
                best = {"combined_auc": val_m["combined_auc"], "epoch": epoch,
                        "state": copy.deepcopy(model.state_dict())}
            elif epoch - best["epoch"] >= cfg.early_stopping_patience:
                print(f"[train] early stop: no combined-AUC gain for {cfg.early_stopping_patience} epochs")
                break

        # Restore the best checkpoint (by validation combined AUC) before logging the artifact.
        model.load_state_dict(best["state"])
        final_val = evaluate(model, val_loader, loss_bce, loss_mse, cfg,
                             data_stats["long_dwell_threshold_z"], device)
        final_train_loss = next(h["train_joint_loss"] for h in history if h["epoch"] == best["epoch"])
        final_metrics = {
            "final_auc_saves": final_val["auc_saves"],
            "final_auc_dwell": final_val["auc_dwell"],
            "final_auc_shares": final_val["auc_shares"],
            "final_combined_auc": final_val["combined_auc"],
            "final_val_joint_loss": final_val["val_joint_loss"],
            "final_train_joint_loss": final_train_loss,
            "final_rmse_dwell": final_val["rmse_dwell"],
            "best_epoch": best["epoch"],
            "epochs_trained": len(history),
            "negative_transfer_detected": int(monitor.ever_flagged),
        }
        mlflow.log_metrics(final_metrics)

        with tempfile.TemporaryDirectory() as tmp:
            curves = os.path.join(tmp, "training_curves.png")
            plot_training_curves(history, curves, best["epoch"])
            mlflow.log_artifact(curves)
            hist_path = os.path.join(tmp, "epoch_history.json")
            with open(hist_path, "w") as fh:
                json.dump(history, fh, indent=2)
            mlflow.log_artifact(hist_path)
        plot_training_curves(history, "training_curves.png", best["epoch"])  # local copy for the write-up

        # --- Log the trained model as an MLflow artifact ---
        example = next(iter(val_loader))[0][:8].cpu()
        model_info = log_pytorch_model(model, example)
        print(f"[mlflow] model logged: {model_info.model_uri}")

    # --- Registration in the Azure ML model registry ---
    registered = None
    if ml_client is not None:
        try:
            registered = register_model_in_azure(ml_client, run_id, model_info.model_uri, final_val)
            MlflowClient().set_tag(run_id, "azureml_model_name", registered.name)
            MlflowClient().set_tag(run_id, "azureml_model_version", str(registered.version))
        except ClientAuthenticationError:
            print("[azure] Authentication failed - run `az login` in your terminal, then re-run.")
        except Exception as exc:
            print(f"[azure] Registration failed: {exc}")

    summary = {
        "mlflow_run_id": run_id,
        "mlflow_experiment": EXPERIMENT_NAME,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "mlflow_model_uri": f"runs:/{run_id}/model",
        "tracking_env": tracking_env,
        "azureml_model_name": registered.name if registered else None,
        "azureml_model_version": str(registered.version) if registered else None,
        "azureml_model_id": getattr(registered, "id", None) if registered else None,
        "metrics": final_metrics,
        "history": history,
        "data": data_stats,
        "config": asdict(cfg),
        "n_parameters": n_params,
        "versions": {"torch": torch.__version__, "mlflow": mlflow.__version__},
        "device": str(device),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(SUMMARY_FILE, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 64)
    print("VALIDATION METRICS (registered checkpoint, epoch %d)" % best["epoch"])
    print(f"  AUC Saves           {final_val['auc_saves']:.4f}")
    print(f"  AUC Dwell Time      {final_val['auc_dwell']:.4f}  (long-dwell >= median)")
    print(f"  AUC Private Shares  {final_val['auc_shares']:.4f}")
    print(f"  Combined AUC        {final_val['combined_auc']:.4f}  (mean of the three)")
    print(f"  Final joint loss    {final_val['val_joint_loss']:.4f}  val / {final_train_loss:.4f} train")
    print("-" * 64)
    print(f"  MLflow Run ID:                 {run_id}")
    if registered:
        print(f"  Azure ML registered model:     {registered.name}")
        print(f"  Azure ML model version:        {registered.version}")
    else:
        print("  Azure ML registered model:     (not registered - see [azure] messages above)")
    print("=" * 64)
    return summary


if __name__ == "__main__":
    main()
