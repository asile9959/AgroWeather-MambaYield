"""
Deep weather-yield experiment with MSTC + LSTM-Attention + Mamba-compatible branch.

Run with the PyTorch environment on this machine:
    D:\\anaconda3\\envs\\cama\\python.exe weather_yield_deep_experiment.py

The script reuses the alignment and weather feature engineering utilities from
weather_yield_experiment.py, then builds yearly sequences per crop/state/season.
If mamba_ssm is unavailable, a residual feed-forward temporal block is used as a
Mamba-compatible fallback so the experiment still runs. The model also includes
an ECA-style weather-variable attention block for factor-importance analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from weather_yield_experiment import (
    ExperimentConfig,
    add_annual_weather_features,
    add_group_lags,
    build_feature_columns,
    load_and_align,
)


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


@dataclass
class Sample:
    seq_x: np.ndarray
    knowledge_x: np.ndarray
    y: float
    crop_idx: int
    stage_idx: int


class YieldSequenceDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "seq_x": torch.tensor(s.seq_x, dtype=torch.float32),
            "knowledge_x": torch.tensor(s.knowledge_x, dtype=torch.float32),
            "y": torch.tensor([s.y], dtype=torch.float32),
            "crop_idx": torch.tensor(s.crop_idx, dtype=torch.long),
            "stage_idx": torch.tensor(s.stage_idx, dtype=torch.long),
        }


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(seq_h)
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(alpha * seq_h, dim=1)
        return context, alpha


class WeatherVariableECA(nn.Module):
    """ECA-style attention over weather variables in a [batch, time, feature] sequence."""

    def __init__(self, feature_dim: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd to preserve feature length.")
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.feature_dim = feature_dim

    def forward(self, seq_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor = seq_x.mean(dim=1).unsqueeze(1)
        weights = torch.sigmoid(self.conv(descriptor)).squeeze(1)
        return seq_x * weights.unsqueeze(1), weights


class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution block for local weather-process patterns."""

    def __init__(self, hidden_dim: int, dropout: float, dilations: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                )
                for dilation in dilations
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        fused = torch.stack([branch(x_t) for branch in self.branches], dim=0).mean(dim=0)
        fused = self.dropout(fused).transpose(1, 2)
        return self.norm(x + fused)


class FallbackMambaBlock(nn.Module):
    """Small residual temporal block used when mamba_ssm is not installed."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


def build_mamba_block(
    hidden_dim: int,
    dropout: float,
    expand: int = 2,
    variant: str = "auto",
) -> tuple[nn.Module, str]:
    if variant in {"auto", "mamba2"}:
        try:
            from mamba_ssm import Mamba2  # type: ignore

            class RealMamba2Wrapper(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.mamba = Mamba2(d_model=hidden_dim, d_state=64, d_conv=4, expand=expand)
                    self.drop = nn.Dropout(dropout)
                    self.norm = nn.LayerNorm(hidden_dim)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.norm(x + self.drop(self.mamba(x)))

            return RealMamba2Wrapper(), "mamba_ssm.Mamba2"
        except Exception:
            if variant == "mamba2":
                return FallbackMambaBlock(hidden_dim, dropout), "fallback_residual_temporal_block"

    if variant in {"auto", "mamba"}:
        try:
            from mamba_ssm import Mamba  # type: ignore

            class RealMambaWrapper(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=expand)
                    self.drop = nn.Dropout(dropout)
                    self.norm = nn.LayerNorm(hidden_dim)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.norm(x + self.drop(self.mamba(x)))

            return RealMambaWrapper(), "mamba_ssm.Mamba"
        except Exception:
            pass

    return FallbackMambaBlock(hidden_dim, dropout), "fallback_residual_temporal_block"


class BranchAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h_lstm: torch.Tensor, h_mamba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(torch.cat([h_lstm, h_mamba], dim=-1)), dim=-1)
        fused = weights[:, 0:1] * h_lstm + weights[:, 1:2] * h_mamba
        return fused, weights


class KnowledgeDrivenYieldModel(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        knowledge_dim: int,
        crop_vocab: int,
        stage_vocab: int,
        hidden_dim: int = 64,
        emb_dim: int = 8,
        dropout: float = 0.1,
        mamba_variant: str = "auto",
        use_mstc: bool = True,
        use_variable_attention: bool = True,
        use_mamba_branch: bool = True,
    ):
        super().__init__()
        self.use_variable_attention = use_variable_attention
        self.use_mstc = use_mstc
        self.use_mamba_branch = use_mamba_branch
        self.variable_attention = WeatherVariableECA(seq_dim) if use_variable_attention else None
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.mstc = MultiScaleTemporalConv(hidden_dim, dropout) if use_mstc else None

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_attn = TemporalAttention(hidden_dim)

        if use_mamba_branch:
            self.mamba_block, self.mamba_impl = build_mamba_block(hidden_dim, dropout, variant=mamba_variant)
            self.mamba_attn = TemporalAttention(hidden_dim)
            self.branch_fusion = BranchAttentionFusion(hidden_dim)
        else:
            self.mamba_block = None
            self.mamba_attn = None
            self.branch_fusion = None
            self.mamba_impl = "disabled"

        self.crop_emb = nn.Embedding(max(crop_vocab, 1), emb_dim)
        self.stage_emb = nn.Embedding(max(stage_vocab, 1), emb_dim)

        self.knowledge_encoder = nn.Sequential(
            nn.Linear(knowledge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.symbolic_head = nn.Linear(knowledge_dim, 1)

        fusion_dim = hidden_dim + hidden_dim + emb_dim + emb_dim + 1
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        seq_x: torch.Tensor,
        knowledge_x: torch.Tensor,
        crop_idx: torch.Tensor,
        stage_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.variable_attention is not None:
            seq_x, variable_weights = self.variable_attention(seq_x)
        else:
            variable_weights = torch.ones(seq_x.size(0), seq_x.size(-1), device=seq_x.device)

        x = self.seq_proj(seq_x)
        if self.mstc is not None:
            x = self.mstc(x)

        lstm_out, _ = self.lstm(x)
        h_lstm, alpha_lstm = self.lstm_attn(lstm_out)

        if self.mamba_block is not None and self.mamba_attn is not None and self.branch_fusion is not None:
            mamba_out = self.mamba_block(x)
            h_mamba, alpha_mamba = self.mamba_attn(mamba_out)
            h_time, branch_weights = self.branch_fusion(h_lstm, h_mamba)
        else:
            h_time = h_lstm
            alpha_mamba = torch.zeros_like(alpha_lstm)
            branch_weights = torch.zeros(seq_x.size(0), 2, device=seq_x.device)
            branch_weights[:, 0] = 1.0

        h_knowledge = self.knowledge_encoder(knowledge_x)
        symbolic_pred = self.symbolic_head(knowledge_x)
        h_crop = self.crop_emb(crop_idx)
        h_stage = self.stage_emb(stage_idx)

        fusion = torch.cat([h_time, h_knowledge, h_crop, h_stage, symbolic_pred], dim=-1)
        pred = self.regressor(fusion)

        aux = {
            "symbolic_pred": symbolic_pred,
            "alpha_lstm": alpha_lstm,
            "alpha_mamba": alpha_mamba,
            "branch_weights": branch_weights,
            "variable_weights": variable_weights,
        }
        return pred, aux


def split_samples_by_group(
    df: pd.DataFrame,
    seq_cols: list[str],
    knowledge_cols: list[str],
    target_col: str,
    crop_col: str,
    stage_col: str,
    group_cols: list[str],
    window_size: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    test_samples: list[Sample] = []

    for _, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("year").reset_index(drop=True)
        group_samples: list[Sample] = []
        if len(g) <= window_size:
            continue
        for i in range(window_size, len(g)):
            hist = g.iloc[i - window_size : i]
            cur = g.iloc[i]
            group_samples.append(
                Sample(
                    seq_x=hist[seq_cols].to_numpy(dtype=np.float32),
                    knowledge_x=cur[knowledge_cols].to_numpy(dtype=np.float32),
                    y=np.float32(cur[target_col]),
                    crop_idx=int(cur[crop_col]),
                    stage_idx=int(cur[stage_col]),
                )
            )

        n = len(group_samples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        train_samples.extend(group_samples[:n_train])
        val_samples.extend(group_samples[n_train : n_train + n_val])
        test_samples.extend(group_samples[n_train + n_val :])

    return train_samples, val_samples, test_samples


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    denom = max(float(np.sum(np.abs(y_true))), 1e-8)
    wmape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100)
    return {"RMSE": rmse, "MAE": mae, "WMAPE": wmape, "R2": r2}


def train_model(
    model: KnowledgeDrivenYieldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    lambda_symbolic: float,
    device: torch.device,
) -> list[dict[str, float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            seq_x = batch["seq_x"].to(device)
            knowledge_x = batch["knowledge_x"].to(device)
            y = batch["y"].to(device)
            crop_idx = batch["crop_idx"].to(device)
            stage_idx = batch["stage_idx"].to(device)

            pred, aux = model(seq_x, knowledge_x, crop_idx, stage_idx)
            pred_loss = mse(pred, y)
            symbolic_loss = mse(pred.detach(), aux["symbolic_pred"])
            loss = pred_loss + lambda_symbolic * symbolic_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred, aux = model(
                    batch["seq_x"].to(device),
                    batch["knowledge_x"].to(device),
                    batch["crop_idx"].to(device),
                    batch["stage_idx"].to(device),
                )
                y = batch["y"].to(device)
                val_losses.append(float(mse(pred, y).cpu()))

        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "val_mse": float(np.mean(val_losses)) if val_losses else np.nan,
        }
        history.append(row)
        print(f"epoch {epoch + 1:03d}: train_loss={row['train_loss']:.6f} val_mse={row['val_mse']:.6f}")

    return history


def predict(
    model: KnowledgeDrivenYieldModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    branches = []
    variable_weights = []
    with torch.no_grad():
        for batch in loader:
            pred, aux = model(
                batch["seq_x"].to(device),
                batch["knowledge_x"].to(device),
                batch["crop_idx"].to(device),
                batch["stage_idx"].to(device),
            )
            preds.append(pred.cpu().numpy().reshape(-1))
            trues.append(batch["y"].cpu().numpy().reshape(-1))
            branches.append(aux["branch_weights"].cpu().numpy())
            variable_weights.append(aux["variable_weights"].cpu().numpy())
    return (
        np.concatenate(trues),
        np.concatenate(preds),
        np.concatenate(branches),
        np.concatenate(variable_weights),
    )


def prepare_panel(config: ExperimentConfig) -> tuple[pd.DataFrame, list[str], list[str], dict[str, int]]:
    df, _ = load_and_align(config)
    df, _ = add_annual_weather_features(df, config)
    df = add_group_lags(df, config)
    _, _, weather_focus_cols = build_feature_columns(df, config)

    stage_source = config.season_col if config.season_col in df.columns else config.crop_col
    df = df.dropna(subset=weather_focus_cols + [config.target_col, config.crop_col, stage_source]).copy()
    df[config.crop_col] = df[config.crop_col].astype(str)
    df["stage_proxy"] = df[stage_source].astype(str)

    crop_encoder = LabelEncoder()
    stage_encoder = LabelEncoder()
    df["crop_idx"] = crop_encoder.fit_transform(df[config.crop_col])
    df["stage_idx"] = stage_encoder.fit_transform(df["stage_proxy"])

    # Use only weather-engineered variables as the sequence. This keeps the
    # experiment focused on weather-to-yield influence rather than management
    # or target leakage from production.
    seq_cols = weather_focus_cols
    knowledge_cols = weather_focus_cols

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    df[seq_cols] = scaler_x.fit_transform(df[seq_cols])
    df[knowledge_cols] = df[seq_cols]
    df[config.target_col] = scaler_y.fit_transform(df[[config.target_col]]).reshape(-1)
    df.attrs["target_scaler_mean"] = float(scaler_y.mean_[0])
    df.attrs["target_scaler_scale"] = float(scaler_y.scale_[0])

    vocabs = {
        "crop_vocab": int(df["crop_idx"].nunique()),
        "stage_vocab": int(df["stage_idx"].nunique()),
    }
    return df, seq_cols, knowledge_cols, vocabs


def inverse_target(values: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    return values * df.attrs["target_scaler_scale"] + df.attrs["target_scaler_mean"]


def run_deep_experiment(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(
        yield_path=args.yield_path,
        weather_path=args.weather_path,
        soil_path=args.soil_path,
        out_dir=out_dir,
        target_col=args.target_col,
        tbase=args.tbase,
        period_days=args.period_days,
    )

    print("Preparing aligned weather-yield panel...", flush=True)
    df, seq_cols, knowledge_cols, vocabs = prepare_panel(config)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sort_values(["crop", "state", "season", "year"]).head(args.max_rows).copy()
        print(f"Using first {len(df)} rows for a smoke experiment.", flush=True)
    group_cols = [c for c in [config.crop_col, config.state_col, config.season_col] if c in df.columns]

    print("Building sequence samples...", flush=True)
    train_samples, val_samples, test_samples = split_samples_by_group(
        df=df,
        seq_cols=seq_cols,
        knowledge_cols=knowledge_cols,
        target_col=config.target_col,
        crop_col="crop_idx",
        stage_col="stage_idx",
        group_cols=group_cols,
        window_size=args.window_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    if not train_samples or not test_samples:
        raise ValueError("No train/test sequence samples. Try a smaller --window-size.")

    train_loader = DataLoader(YieldSequenceDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(YieldSequenceDataset(val_samples), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(YieldSequenceDataset(test_samples), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = KnowledgeDrivenYieldModel(
        seq_dim=len(seq_cols),
        knowledge_dim=len(knowledge_cols),
        crop_vocab=vocabs["crop_vocab"],
        stage_vocab=vocabs["stage_vocab"],
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        mamba_variant=args.mamba_variant,
        use_mstc=not args.disable_mstc,
        use_variable_attention=not args.disable_variable_attention,
        use_mamba_branch=not args.disable_mamba_branch,
    )
    print(f"Using device: {device}", flush=True)
    print(f"Mamba implementation: {model.mamba_impl}", flush=True)
    print(
        "Enabled modules: "
        f"MSTC={model.use_mstc}, "
        f"MambaBranch={model.use_mamba_branch}, "
        f"variable_attention={model.use_variable_attention}",
        flush=True,
    )
    print(f"samples train/val/test = {len(train_samples)}/{len(val_samples)}/{len(test_samples)}", flush=True)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        lambda_symbolic=args.lambda_symbolic,
        device=device,
    )

    true_scaled, pred_scaled, branch_weights, variable_weights = predict(model, test_loader, device)
    trues = inverse_target(true_scaled, df)
    preds = inverse_target(pred_scaled, df)
    metric_row = metrics(trues, preds)
    variable_attention_summary = (
        pd.DataFrame(
            {
                "feature": seq_cols,
                "mean_attention": variable_weights.mean(axis=0),
            }
        )
        .sort_values("mean_attention", ascending=False)
        .reset_index(drop=True)
    )

    pd.DataFrame(history).to_csv(out_dir / "deep_training_history.csv", index=False)
    pd.DataFrame([metric_row]).to_csv(out_dir / "deep_test_metrics.csv", index=False)
    pd.DataFrame({"true_yield": trues, "pred_yield": preds}).to_csv(out_dir / "deep_test_predictions.csv", index=False)
    pd.DataFrame(branch_weights, columns=["w_lstm", "w_mamba"]).to_csv(out_dir / "deep_branch_weights.csv", index=False)
    pd.DataFrame(variable_weights, columns=seq_cols).to_csv(
        out_dir / "deep_variable_attention_weights.csv",
        index=False,
    )
    variable_attention_summary.to_csv(out_dir / "deep_variable_attention_summary.csv", index=False)

    if args.plot:
        plt.figure(figsize=(7, 7))
        plt.scatter(trues, preds, s=12, alpha=0.65)
        min_v = min(float(np.min(trues)), float(np.min(preds)))
        max_v = max(float(np.max(trues)), float(np.max(preds)))
        plt.plot([min_v, max_v], [min_v, max_v], color="red", linewidth=1)
        plt.xlabel("True yield")
        plt.ylabel("Predicted yield")
        plt.title("Deep model predicted vs true yield")
        plt.tight_layout()
        plt.savefig(out_dir / "deep_predicted_vs_true.png", dpi=180)
        plt.close()

    notes = {
        "seq_cols": seq_cols,
        "knowledge_cols": knowledge_cols,
        "enabled_modules": {
            "mstc": model.use_mstc,
            "mamba_branch": model.use_mamba_branch,
            "variable_attention": model.use_variable_attention,
            "mamba_variant_requested": args.mamba_variant,
        },
        "mamba_impl": model.mamba_impl,
        "metrics": metric_row,
        "mean_branch_weights": {
            "w_lstm": float(branch_weights[:, 0].mean()),
            "w_mamba": float(branch_weights[:, 1].mean()),
        },
        "mean_variable_attention": dict(
            zip(variable_attention_summary["feature"], variable_attention_summary["mean_attention"].astype(float))
        ),
        "stage_note": "Existing India data uses season as a stage proxy.",
    }
    (out_dir / "deep_experiment_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== Deep test metrics ===")
    print(pd.DataFrame([metric_row]).to_string(index=False))
    print("Mean branch weights [LSTM, Mamba] =", branch_weights.mean(axis=0))
    print("Top weather-variable attention:")
    print(variable_attention_summary.head(8).to_string(index=False))
    print(f"Outputs saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(r"D:\sicau\code\CAMA-Yield\india_dataset-1")
    parser = argparse.ArgumentParser(description="Run deep weather-yield experiment.")
    parser.add_argument("--yield-path", type=Path, default=default_data_dir / "crop_yield.csv")
    parser.add_argument("--weather-path", type=Path, default=default_data_dir / "state_weather_data_1997_2020.csv")
    parser.add_argument("--soil-path", type=Path, default=default_data_dir / "state_soil_data.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_weather_yield_deep"))
    parser.add_argument("--target-col", default="yield")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mamba-variant", choices=["auto", "mamba2", "mamba", "fallback"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-symbolic", type=float, default=0.05)
    parser.add_argument("--tbase", type=float, default=10.0)
    parser.add_argument("--period-days", type=int, default=120)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--disable-mstc", action="store_true")
    parser.add_argument("--disable-mamba-branch", action="store_true")
    parser.add_argument("--disable-variable-attention", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_deep_experiment(parse_args())
