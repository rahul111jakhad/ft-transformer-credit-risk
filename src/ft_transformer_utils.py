"""PyTorch utilities for FT-Transformer training.

Shared between Lending Club and Home Credit FT-Transformer notebooks so the
training loop, dataset wrapping, early stopping, and sklearn-compatible
prediction interface stay identical across datasets.

Key components:
    TabularDataset       — wraps (x_num, x_cat, y) for DataLoader
    EarlyStopper         — patience-based on validation AUC (maximize)
    train_one_epoch      — one epoch with optional AMP
    evaluate             — forward pass on a loader, returns proba + metrics
    FTTransformerWrapper — sklearn-style .predict_proba() so the existing
                           report_performance() helper works unchanged
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Dataset
# =============================================================================

class TabularDataset(Dataset):
    """Wraps numerical, categorical, and target arrays for a DataLoader.

    All arrays are cast to torch.Tensor once at construction. Numerical
    features become float32; categorical features become int64 (long), as
    required by nn.Embedding inside FT-Transformer.
    """

    def __init__(self, x_num, x_cat, y):
        # Accept DataFrame / ndarray / tensor inputs
        if isinstance(x_num, pd.DataFrame):
            x_num = x_num.values
        if isinstance(x_cat, pd.DataFrame):
            x_cat = x_cat.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values

        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        # FT-Transformer expects long-dtype categoricals (they're embedding lookups)
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


def make_dataloaders(
    data,
    batch_size: int = 1024,
    num_workers: int = 0,
):
    """Build train / valid / test DataLoaders from a PreprocessedData object.

    Args:
        data: src.preprocessing.PreprocessedData
        batch_size: int
        num_workers: int; 0 is safest for Colab (avoid forking issues)

    Returns:
        (train_loader, valid_loader, test_loader)
    """
    train_ds = TabularDataset(data.x_num_train, data.x_cat_train, data.y_train)
    valid_ds = TabularDataset(data.x_num_valid, data.x_cat_valid, data.y_valid)
    test_ds = TabularDataset(data.x_num_test, data.x_cat_test, data.y_test)

    common = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **common)
    return train_loader, valid_loader, test_loader


# =============================================================================
# Early stopping
# =============================================================================

@dataclass
class EarlyStopper:
    """Patience-based early stopping on validation AUC (maximize).

    Stashes the best model state internally so it can be restored after
    training completes.
    """

    patience: int = 10
    min_delta: float = 1e-5

    best_score: float = field(default=-float("inf"), init=False)
    best_epoch: int = field(default=-1, init=False)
    best_state: Optional[dict] = field(default=None, init=False)
    epochs_without_improvement: int = field(default=0, init=False)

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Returns True if we should stop training."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            # Detach and clone so subsequent updates don't mutate the snapshot
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load best-AUC state back into the model in-place."""
        if self.best_state is None:
            return
        model.load_state_dict(self.best_state)


# =============================================================================
# Training loop
# =============================================================================

def _move_to_device(batch, device):
    x_num, x_cat, y = batch
    return x_num.to(device, non_blocking=True), x_cat.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> float:
    """One epoch of training. Returns mean batch loss.

    If use_amp is True, runs forward+loss under autocast and scales the
    backward pass. scaler must be provided when use_amp is True.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x_num, x_cat, y = _move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x_num, x_cat).squeeze(-1)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_num, x_cat).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward pass over a loader. Returns (probabilities, targets, auc)."""
    model.eval()
    all_proba = []
    all_y = []

    for batch in loader:
        x_num, x_cat, y = _move_to_device(batch, device)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x_num, x_cat).squeeze(-1)
        else:
            logits = model(x_num, x_cat).squeeze(-1)
        proba = torch.sigmoid(logits.float())
        all_proba.append(proba.cpu().numpy())
        all_y.append(y.cpu().numpy())

    probas = np.concatenate(all_proba)
    targets = np.concatenate(all_y)
    auc = roc_auc_score(targets, probas)
    return probas, targets, auc


def train_ft_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    max_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    pos_weight: Optional[float] = None,
    patience: int = 10,
    use_amp: bool = True,
    device: str = "cuda",
    log_every: int = 1,
    verbose: bool = True,
) -> tuple[nn.Module, pd.DataFrame, EarlyStopper]:
    """Train an FT-Transformer with early stopping on validation AUC.

    Args:
        model: an instantiated FTTransformer (already moved to device).
        train_loader, valid_loader: from make_dataloaders.
        max_epochs: training upper bound.
        learning_rate, weight_decay: AdamW hyperparameters.
        pos_weight: optional positive-class weight for BCEWithLogitsLoss.
            Pass float(neg_count / pos_count) for balanced-weighted BCE.
        patience: epochs without valid-AUC improvement before stopping.
        use_amp: enable mixed-precision forward/backward (CUDA only).
        device: 'cuda' or 'cpu'.
        log_every: print metrics every N epochs.
        verbose: print headers + per-epoch lines.

    Returns:
        (model_with_best_weights, history_df, early_stopper)
    """
    model = model.to(device)

    # Loss: optionally weighted BCE on logits
    if pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    use_amp = use_amp and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    early = EarlyStopper(patience=patience)
    history = []

    if verbose:
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Valid AUC':>10} | {'Best AUC':>10} | {'Patience':>10}")
        print("-" * 64)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, use_amp
        )
        _, _, valid_auc = evaluate(model, valid_loader, device, use_amp)

        should_stop = early.step(valid_auc, model, epoch)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_auc": valid_auc,
            "best_auc": early.best_score,
            "best_epoch": early.best_epoch,
        })

        if verbose and (epoch % log_every == 0 or epoch == max_epochs):
            print(
                f"{epoch:>5d} | {train_loss:>10.4f} | {valid_auc:>10.5f} | "
                f"{early.best_score:>10.5f} | {early.epochs_without_improvement:>4d}/{patience}"
            )

        if should_stop:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch} (best epoch {early.best_epoch}).")
            break

    # Restore best weights
    early.restore_best(model)
    return model, pd.DataFrame(history), early


# =============================================================================
# Sklearn-style wrapper
# =============================================================================

class FTTransformerWrapper:
    """Wraps a trained FT-Transformer to expose .predict_proba() like sklearn.

    Lets the existing src.evaluation.report_performance() helper work without
    modification. The wrapper remembers which features are numeric vs
    categorical so that incoming concatenated DataFrames (or sliced subsets
    from the same preprocessing) can be split correctly.

    Usage:
        wrapper = FTTransformerWrapper(
            model=trained_model,
            num_cols=data.num_cols,
            cat_cols=data.cat_cols,
            device="cuda",
        )
        report = report_performance(
            wrapper,
            data.x_num_train.join(data.x_cat_train), data.y_train,
            ...
        )

    For convenience, FTTransformerWrapper also accepts num+cat as separate
    args via predict_proba_split().
    """

    def __init__(
        self,
        model: nn.Module,
        num_cols: list,
        cat_cols: list,
        device: str = "cuda",
        batch_size: int = 2048,
        use_amp: bool = True,
    ):
        self.model = model.eval()
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.device = device
        self.batch_size = batch_size
        self.use_amp = use_amp and device.startswith("cuda")

    def predict_proba(self, X) -> np.ndarray:
        """Returns a 2-column array [P(0), P(1)] matching sklearn convention.

        Accepts a DataFrame with both num and cat columns present.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FTTransformerWrapper.predict_proba expects a DataFrame "
                "containing both numerical and categorical columns. "
                "Use predict_proba_split(x_num, x_cat) for separate arrays."
            )
        x_num = X[self.num_cols].values
        x_cat = X[self.cat_cols].values
        return self.predict_proba_split(x_num, x_cat)

    def predict_proba_split(self, x_num, x_cat) -> np.ndarray:
        """Returns [P(0), P(1)] columns given separate numeric/categorical inputs."""
        if isinstance(x_num, pd.DataFrame):
            x_num = x_num.values
        if isinstance(x_cat, pd.DataFrame):
            x_cat = x_cat.values

        x_num_t = torch.as_tensor(x_num, dtype=torch.float32, device=self.device)
        x_cat_t = torch.as_tensor(x_cat, dtype=torch.long, device=self.device)

        probas = []
        with torch.no_grad():
            for start in range(0, len(x_num_t), self.batch_size):
                stop = start + self.batch_size
                xn = x_num_t[start:stop]
                xc = x_cat_t[start:stop]
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(xn, xc).squeeze(-1)
                else:
                    logits = self.model(xn, xc).squeeze(-1)
                probas.append(torch.sigmoid(logits.float()).cpu().numpy())

        p1 = np.concatenate(probas)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


# =============================================================================
# Convenience: build the FT-Transformer model with paper defaults
# =============================================================================

def build_ft_transformer(
    n_num_features: int,
    cat_cardinalities: list,
    *,
    n_blocks: int = 3,
    d_token: int = 192,
    attention_n_heads: int = 8,
    attention_dropout: float = 0.2,
    ffn_d_hidden_multiplier: float = 4 / 3,
    ffn_dropout: float = 0.1,
    residual_dropout: float = 0.0,
    d_out: int = 1,
):
    """Construct an FT-Transformer using rtdl_revisiting_models with paper defaults.

    Defaults match the recommended configuration in Gorishniy et al. (NeurIPS 2021),
    "Revisiting Deep Learning Models for Tabular Data".

    The `attention_n_heads` parameter is exposed for flexibility but defaults
    to 8 (the paper's recommendation). d_token must be divisible by
    attention_n_heads. Reduce attention_n_heads (and/or d_token) if you hit
    GPU memory limits during tuning.
    """
    from rtdl_revisiting_models import FTTransformer

    assert d_token % attention_n_heads == 0, (
        f"d_token ({d_token}) must be divisible by attention_n_heads "
        f"({attention_n_heads})."
    )

    model = FTTransformer(
        n_cont_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        n_blocks=n_blocks,
        d_block=d_token,
        attention_n_heads=attention_n_heads,
        attention_dropout=attention_dropout,
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=ffn_d_hidden_multiplier,
        ffn_dropout=ffn_dropout,
        residual_dropout=residual_dropout,
    )
    return model


# =============================================================================
# Misc helpers
# =============================================================================

def set_torch_seed(seed: int = 42) -> None:
    """Seed torch + cuda for deterministic-ish training.

    Note: full determinism requires set_deterministic_algorithms and disabling
    cudnn benchmark — we trade some reproducibility for ~10-20% speedup.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
