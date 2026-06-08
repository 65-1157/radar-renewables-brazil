"""
src/forecaster.py
=================
Probabilistic 15-day operational forecasting for solar irradiance and
hub-height wind speed at 6 Brazilian coastal radar stations.

Architecture
------------
  x(t) = Trend(t) + Seasonal(t) + Noise(t)

  Trend    : centred moving average, 365-day window (edge-padded)
  Seasonal : day-of-year mean profile estimated from 6-yr history
  Noise    : residuals fed into a multi-quantile LSTM

  Quantiles : Q10, Q25, Q50, Q75, Q90
  Loss      : pinball (quantile regression) loss — one head per quantile
  Input window : 60 days of noise residuals (or Optuna-optimised)
  Forecast horizon : 15 days ahead
  Models    : 12 total (6 sites x 2 variables)

Hyperparameter optimisation
----------------------------
  Optuna TPE sampler searches:
    hidden_size  : [32, 64, 96, 128]
    num_layers   : [1, 2, 3]
    dropout      : [0.1, 0.5]
    lr           : [1e-4, 1e-2]  log scale
    batch_size   : [32, 64, 128]
    input_window : [30, 45, 60, 90]
  Objective : mean val pinball loss over all quantiles
  Best params saved alongside model checkpoint.

Python 3.9 compatible — uses Optional[X] not X | Y.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_LABELS: List[str] = ["Q10", "Q25", "Q50", "Q75", "Q90"]
SITES: List[str] = [
    "Salvador",
    "Natal",
    "Fortaleza",
    "Cabo Frio",
    "Ilha Grande",
    "Ilha da Trindade",
]
VARIABLES: List[str] = ["solar", "wind"]

TREND_WINDOW: int = 365
FORECAST_HORIZON: int = 15
MIN_TRAIN_DAYS: int = 730
GAP_THRESHOLD: float = 0.05

# Default hyperparameters (used when Optuna is skipped)
DEFAULT_INPUT_WINDOW: int = 60
DEFAULT_HIDDEN_SIZE: int = 64
DEFAULT_NUM_LAYERS: int = 2
DEFAULT_DROPOUT: float = 0.2
DEFAULT_LR: float = 1e-3
DEFAULT_BATCH_SIZE: int = 64


# ---------------------------------------------------------------------------
# Helper — pinball loss
# ---------------------------------------------------------------------------

def pinball_loss_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, quantile * err, (quantile - 1) * err)))


# ---------------------------------------------------------------------------
# Signal decomposition
# ---------------------------------------------------------------------------

class SignalDecomposer:
    def __init__(self, trend_window: int = TREND_WINDOW) -> None:
        self.trend_window = trend_window
        self._seasonal_profile: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self, series: pd.Series) -> "SignalDecomposer":
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series must have a DatetimeIndex.")

        self._series = series.copy()
        hw = self.trend_window // 2

        raw_trend = series.rolling(
            window=self.trend_window, center=True, min_periods=hw
        ).mean()
        trend = raw_trend.fillna(method="ffill").fillna(method="bfill")
        self._trend = trend

        detrended = series - trend

        doy = series.index.dayofyear
        profile = np.zeros(367)
        counts = np.zeros(367)
        for i, d in enumerate(doy):
            profile[d] += detrended.iloc[i]
            counts[d] += 1
        with np.errstate(invalid="ignore"):
            profile = np.where(counts > 0, profile / counts, 0.0)
        smooth = (
            pd.Series(profile[1:])
            .rolling(15, center=True, min_periods=1)
            .mean()
        )
        self._seasonal_profile = np.concatenate([[0.0], smooth.values])

        seasonal_vals = np.array([self._seasonal_profile[d] for d in doy])
        residual = detrended - seasonal_vals
        self._residual = pd.Series(
            residual.values, index=series.index, name="residual"
        )

        self._is_fitted = True
        return self

    def trend_at(self, dates: pd.DatetimeIndex) -> np.ndarray:
        self._check_fitted()
        t_vals = self._trend.reindex(dates)
        missing = t_vals.isna()
        if missing.any():
            tail = self._trend.dropna().tail(90)
            x = np.arange(len(tail))
            coeffs = np.polyfit(x, tail.values, 1)
            slope, intercept = coeffs
            last_idx = len(tail) - 1
            for i, dt in enumerate(dates[missing]):
                gap = (dt - tail.index[-1]).days
                t_vals[dt] = slope * (last_idx + gap) + intercept
        return t_vals.values

    def seasonal_at(self, dates: pd.DatetimeIndex) -> np.ndarray:
        self._check_fitted()
        doys = dates.dayofyear
        return np.array([self._seasonal_profile[d] for d in doys])

    @property
    def residuals(self) -> pd.Series:
        self._check_fitted()
        return self._residual

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before using this decomposer.")


# ---------------------------------------------------------------------------
# Phase 1 — Empirical quantile forecaster
# ---------------------------------------------------------------------------

class EmpiricalQuantileForecaster:
    def __init__(self, decomposer: SignalDecomposer) -> None:
        self._dc = decomposer
        self._doy_quantiles: Dict[int, Dict[float, float]] = {}
        self._is_fitted: bool = False

    def fit(self) -> "EmpiricalQuantileForecaster":
        resid = self._dc.residuals
        doys = resid.index.dayofyear
        for doy in range(1, 367):
            vals = resid.values[doys == doy]
            if len(vals) == 0:
                vals = resid.values[doys == 365]
            self._doy_quantiles[doy] = {
                q: float(np.quantile(vals, q)) for q in QUANTILES
            }
        self._is_fitted = True
        return self

    def forecast(
        self,
        anchor_date: pd.Timestamp,
        n_days: int = FORECAST_HORIZON,
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first.")

        future_dates = pd.date_range(
            start=anchor_date + pd.Timedelta(days=1), periods=n_days, freq="D"
        )
        trend = self._dc.trend_at(future_dates)
        seasonal = self._dc.seasonal_at(future_dates)
        doys = future_dates.dayofyear

        rows = []
        for i, doy in enumerate(doys):
            base = trend[i] + seasonal[i]
            q_dict = self._doy_quantiles[doy]
            row = {
                lbl: base + q_dict[q]
                for q, lbl in zip(QUANTILES, QUANTILE_LABELS)
            }
            row["date"] = future_dates[i]
            rows.append(row)

        return pd.DataFrame(rows).set_index("date").clip(lower=0.0)


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------

class PersistenceForecaster:
    """Naive baseline: last observed value repeated for all horizons."""

    def __init__(self, decomposer: SignalDecomposer) -> None:
        self._dc = decomposer

    def forecast(
        self,
        anchor_date: pd.Timestamp,
        n_days: int = FORECAST_HORIZON,
    ) -> pd.DataFrame:
        resid = self._dc.residuals
        last_resid = float(resid[resid.index <= anchor_date].iloc[-1])

        future_dates = pd.date_range(
            start=anchor_date + pd.Timedelta(days=1), periods=n_days, freq="D"
        )
        trend = self._dc.trend_at(future_dates)
        seasonal = self._dc.seasonal_at(future_dates)

        rows = []
        for i, dt in enumerate(future_dates):
            base = max(0.0, trend[i] + seasonal[i] + last_resid)
            row = {lbl: base for lbl in QUANTILE_LABELS}
            row["date"] = dt
            rows.append(row)

        return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# LSTM network definition (standalone so Optuna can reinstantiate it)
# ---------------------------------------------------------------------------

def _build_net(
    hidden_size: int,
    num_layers: int,
    dropout: float,
    n_quantiles: int,
    horizon: int,
):
    """Build and return a fresh LSTM network."""
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, n_quantiles * horizon)
            self.nq = n_quantiles
            self.h = horizon

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            raw = self.head(last)
            return raw.view(-1, self.nq, self.h)

    return _Net()


# ---------------------------------------------------------------------------
# Phase 2 — Multi-quantile LSTM
# ---------------------------------------------------------------------------

def _try_import_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class _QuantileLSTMModel:
    """
    Multi-quantile LSTM with Optuna hyperparameter search,
    train/val loss tracking, gap-based early stopping,
    SHAP global feature importance, and LIME local explanations.
    """

    def __init__(
        self,
        input_window: int = DEFAULT_INPUT_WINDOW,
        horizon: int = FORECAST_HORIZON,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        if not _try_import_torch():
            raise ImportError(
                "PyTorch is required. Install with: pip install torch"
            )
        import torch

        self.input_window = input_window
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_quantiles = len(QUANTILES)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._net = _build_net(
            hidden_size, num_layers, dropout, self.n_quantiles, horizon
        ).to(self._device)

        self._scaler_mean: float = 0.0
        self._scaler_std: float = 1.0
        self._is_trained: bool = False
        self._X_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_params: Dict = {}

    # ------------------------------------------------------------------
    def _pinball_loss_torch(self, preds, targets):
        import torch
        total = torch.zeros(1, device=self._device)
        for qi, q in enumerate(QUANTILES):
            err = targets - preds[:, qi, :]
            loss = torch.where(err >= 0, q * err, (q - 1) * err)
            total = total + loss.mean()
        return total / len(QUANTILES)

    def _make_sequences(
        self, residuals: np.ndarray, input_window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        T = input_window
        H = self.horizon
        n = len(residuals)
        for start in range(n - T - H + 1):
            X.append(residuals[start : start + T])
            Y.append(residuals[start + T : start + T + H])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    # ------------------------------------------------------------------
    def tune(
        self,
        residuals: np.ndarray,
        n_trials: int = 20,
        val_frac: float = 0.15,
        epochs_per_trial: int = 30,
        patience_per_trial: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        Run Optuna hyperparameter search.

        Searches over:
          hidden_size  : [32, 64, 96, 128]
          num_layers   : [1, 2, 3]
          dropout      : [0.1, 0.5]
          lr           : [1e-4, 1e-2]  log scale
          batch_size   : [32, 64, 128]
          input_window : [30, 45, 60, 90]

        Returns best_params dict.
        Updates self.input_window and rebuilds self._net with best params
        so that fit() can be called immediately after.
        """
        try:
            import optuna
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("Install with: pip install optuna torch")

        # Silence Optuna's per-trial output unless verbose
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        scaler_mean = float(np.mean(residuals))
        scaler_std = float(np.std(residuals)) or 1.0
        scaled = (residuals - scaler_mean) / scaler_std

        def objective(trial: optuna.Trial) -> float:
            h_size = trial.suggest_categorical(
                "hidden_size", [32, 64, 96, 128]
            )
            n_layers = trial.suggest_int("num_layers", 1, 3)
            drop = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            b_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            i_win = trial.suggest_categorical(
                "input_window", [30, 45, 60, 90]
            )

            X, Y = self._make_sequences(scaled, i_win)
            if len(X) < 10:
                return float("inf")

            n_val = max(1, int(len(X) * val_frac))
            X_tr, Y_tr = X[:-n_val], Y[:-n_val]
            X_val, Y_val = X[-n_val:], Y[-n_val:]

            X_tr_t = torch.tensor(X_tr[:, :, None]).to(self._device)
            Y_tr_t = torch.tensor(Y_tr).to(self._device)
            X_val_t = torch.tensor(X_val[:, :, None]).to(self._device)
            Y_val_t = torch.tensor(Y_val).to(self._device)

            net = _build_net(
                h_size, n_layers, drop, self.n_quantiles, self.horizon
            ).to(self._device)
            optim = torch.optim.Adam(net.parameters(), lr=lr)

            ds = TensorDataset(X_tr_t, Y_tr_t)
            loader = DataLoader(ds, batch_size=b_size, shuffle=True)

            best_val = math.inf
            patience_counter = 0

            for epoch in range(epochs_per_trial):
                net.train()
                for xb, yb in loader:
                    optim.zero_grad()
                    preds = net(xb)
                    loss = self._pinball_loss_torch(
                        preds, yb.unsqueeze(1).expand_as(preds)
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optim.step()

                net.eval()
                with torch.no_grad():
                    val_preds = net(X_val_t)
                    val_loss = self._pinball_loss_torch(
                        val_preds,
                        Y_val_t.unsqueeze(1).expand_as(val_preds),
                    )
                vl = float(val_loss.item())

                # Optuna pruning
                trial.report(vl, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if vl < best_val:
                    best_val = vl
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience_per_trial:
                        break

            return best_val

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

        best = study.best_params
        self.best_params = best
        logger.info("Optuna best params: %s", best)
        logger.info("Optuna best val pinball: %.5f", study.best_value)

        # Rebuild model with best params
        self.input_window = best["input_window"]
        self.hidden_size = best["hidden_size"]
        self.num_layers = best["num_layers"]
        self.dropout = best["dropout"]
        self._net = _build_net(
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.n_quantiles,
            self.horizon,
        ).to(self._device)

        return best

    # ------------------------------------------------------------------
    def fit(
        self,
        residuals: np.ndarray,
        epochs: int = 100,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_frac: float = 0.15,
        patience: int = 10,
        gap_threshold: float = GAP_THRESHOLD,
        verbose: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        Train on residuals using current hyperparameters.
        Call tune() first for Optuna-optimised params, or use defaults.

        Returns (train_losses, val_losses) per epoch.
        Early stopping on patience AND train/val gap.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Use best lr/batch from Optuna if available
        if self.best_params:
            lr = self.best_params.get("lr", lr)
            batch_size = self.best_params.get("batch_size", batch_size)

        self._scaler_mean = float(np.mean(residuals))
        self._scaler_std = float(np.std(residuals)) or 1.0
        scaled = (residuals - self._scaler_mean) / self._scaler_std

        X, Y = self._make_sequences(scaled, self.input_window)
        n_val = max(1, int(len(X) * val_frac))
        X_tr, Y_tr = X[:-n_val], Y[:-n_val]
        X_val, Y_val = X[-n_val:], Y[-n_val:]

        self._X_train = X_tr.copy()
        self._X_val = X_val.copy()

        X_tr_t = torch.tensor(X_tr[:, :, None]).to(self._device)
        Y_tr_t = torch.tensor(Y_tr).to(self._device)
        X_val_t = torch.tensor(X_val[:, :, None]).to(self._device)
        Y_val_t = torch.tensor(Y_val).to(self._device)

        ds = TensorDataset(X_tr_t, Y_tr_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(self._net.parameters(), lr=lr)

        self.train_losses = []
        self.val_losses = []
        best_val = math.inf
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self._net.train()
            batch_losses = []
            for xb, yb in loader:
                optim.zero_grad()
                preds = self._net(xb)
                loss = self._pinball_loss_torch(
                    preds, yb.unsqueeze(1).expand_as(preds)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optim.step()
                batch_losses.append(float(loss.item()))
            train_loss = float(np.mean(batch_losses))
            self.train_losses.append(train_loss)

            self._net.eval()
            with torch.no_grad():
                val_preds = self._net(X_val_t)
                val_loss = self._pinball_loss_torch(
                    val_preds, Y_val_t.unsqueeze(1).expand_as(val_preds)
                )
            vl = float(val_loss.item())
            self.val_losses.append(vl)

            gap = vl - train_loss

            if verbose and epoch % 10 == 0:
                logger.info(
                    "Epoch %3d/%d  train=%.5f  val=%.5f  gap=%.5f",
                    epoch, epochs, train_loss, vl, gap,
                )

            if vl < best_val:
                best_val = vl
                patience_counter = 0
                best_state = {
                    k: v.clone()
                    for k, v in self._net.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stop (patience=%d) at epoch %d",
                        patience, epoch,
                    )
                    break

            if gap > gap_threshold:
                logger.info(
                    "Early stop (gap=%.4f > %.4f) at epoch %d",
                    gap, gap_threshold, epoch,
                )
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._is_trained = True
        return self.train_losses, self.val_losses

    # ------------------------------------------------------------------
    def predict(self, last_residuals: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Call .fit() before .predict().")
        import torch

        scaled = (last_residuals - self._scaler_mean) / self._scaler_std
        x = torch.tensor(
            scaled[None, :, None], dtype=torch.float32
        ).to(self._device)
        self._net.eval()
        with torch.no_grad():
            out = self._net(x)
        return out.squeeze(0).cpu().numpy() * self._scaler_std + self._scaler_mean

    # ------------------------------------------------------------------
    def compute_shap(
        self,
        n_background: int = 100,
        n_explain: int = 50,
    ) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Call .fit() before compute_shap().")
        if self._X_train is None or self._X_val is None:
            raise RuntimeError("No training data stored — retrain the model.")
        try:
            import shap
            import torch
        except ImportError:
            raise ImportError("Install with: pip install shap")

        n_bg = min(n_background, len(self._X_train))
        idx_bg = np.random.choice(len(self._X_train), n_bg, replace=False)
        background = torch.tensor(
            self._X_train[idx_bg, :, None], dtype=torch.float32
        ).to(self._device)

        n_exp = min(n_explain, len(self._X_val))
        X_exp = torch.tensor(
            self._X_val[-n_exp:, :, None], dtype=torch.float32
        ).to(self._device)

        import torch.nn as nn

        class _FlatNet(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                out = self.net(x)
                return out.view(out.shape[0], -1)

        flat_net = _FlatNet(self._net).to(self._device)
        flat_net.eval()

        explainer = shap.DeepExplainer(flat_net, background)
        shap_vals = explainer.shap_values(X_exp)
        stacked = np.stack(
            [sv[:, :, 0] for sv in shap_vals], axis=-1
        )
        self.shap_values = stacked
        shap_mean_abs = np.mean(np.abs(stacked), axis=(0, 2))
        self.shap_importance = shap_mean_abs
        logger.info(
            "SHAP top-3 lags: %s",
            np.argsort(shap_mean_abs)[::-1][:3].tolist(),
        )
        return shap_mean_abs

    # ------------------------------------------------------------------
    def explain_forecast_lime(
        self,
        last_residuals: np.ndarray,
        quantile_idx: int = 2,
        horizon_idx: int = 0,
        num_features: int = 10,
    ) -> object:
        if not self._is_trained:
            raise RuntimeError("Call .fit() before explain_forecast_lime().")
        if self._X_train is None:
            raise RuntimeError("No training data stored — retrain the model.")
        try:
            import torch
            from lime import lime_tabular
        except ImportError:
            raise ImportError("Install with: pip install lime")

        output_idx = quantile_idx * self.horizon + horizon_idx

        def _predict_fn(X_flat: np.ndarray) -> np.ndarray:
            X_3d = X_flat.reshape(
                -1, self.input_window, 1
            ).astype(np.float32)
            t = torch.tensor(X_3d).to(self._device)
            self._net.eval()
            with torch.no_grad():
                out = self._net(t)
                flat = out.view(out.shape[0], -1)
            return flat[:, output_idx].cpu().numpy()

        feature_names = [f"lag_{i+1}d" for i in range(self.input_window)]

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=self._X_train,
            feature_names=feature_names,
            mode="regression",
            verbose=False,
        )

        scaled_input = (
            (last_residuals - self._scaler_mean) / self._scaler_std
        )
        explanation = explainer.explain_instance(
            scaled_input,
            _predict_fn,
            num_features=num_features,
        )
        logger.info(
            "LIME top feature: %s", explanation.as_list()[0][0]
        )
        return explanation

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        import torch
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "scaler_mean": self._scaler_mean,
                "scaler_std": self._scaler_std,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_params": self.best_params,
                "input_window": self.input_window,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            path,
        )
        logger.info("LSTM saved -> %s", path)

    def load(self, path: Path) -> None:
        import torch
        ckpt = torch.load(path, map_location=self._device)
        self.input_window = ckpt.get("input_window", DEFAULT_INPUT_WINDOW)
        self.hidden_size = ckpt.get("hidden_size", DEFAULT_HIDDEN_SIZE)
        self.num_layers = ckpt.get("num_layers", DEFAULT_NUM_LAYERS)
        self.dropout = ckpt.get("dropout", DEFAULT_DROPOUT)
        self._net = _build_net(
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.n_quantiles,
            self.horizon,
        ).to(self._device)
        self._net.load_state_dict(ckpt["state_dict"])
        self._scaler_mean = ckpt["scaler_mean"]
        self._scaler_std = ckpt["scaler_std"]
        self.train_losses = ckpt.get("train_losses", [])
        self.val_losses = ckpt.get("val_losses", [])
        self.best_params = ckpt.get("best_params", {})
        self._is_trained = True
        logger.info("LSTM loaded <- %s", path)


# ---------------------------------------------------------------------------
# Solar-wind correlation
# ---------------------------------------------------------------------------

class SiteCorrelationEstimator:
    SUMMER_MONTHS = {11, 12, 1, 2, 3}
    WINTER_MONTHS = {6, 7, 8}

    def __init__(self) -> None:
        self.correlations: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _season(month: int) -> str:
        if month in SiteCorrelationEstimator.SUMMER_MONTHS:
            return "summer"
        if month in SiteCorrelationEstimator.WINTER_MONTHS:
            return "winter"
        return "shoulder"

    def fit(
        self,
        solar_resid: pd.Series,
        wind_resid: pd.Series,
        site: str,
    ) -> "SiteCorrelationEstimator":
        common_idx = solar_resid.index.intersection(wind_resid.index)
        s = solar_resid.loc[common_idx]
        w = wind_resid.loc[common_idx]
        months = common_idx.month

        self.correlations[site] = {}
        for season in ("summer", "winter", "shoulder"):
            mask = np.array([self._season(m) == season for m in months])
            if mask.sum() < 10:
                self.correlations[site][season] = 0.0
                continue
            sv, wv = s.values[mask], w.values[mask]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = float(np.corrcoef(sv, wv)[0, 1])
            self.correlations[site][season] = 0.0 if np.isnan(rho) else rho
        return self

    def get(self, site: str, month: int) -> float:
        season = self._season(month)
        return self.correlations.get(site, {}).get(season, 0.0)


# ---------------------------------------------------------------------------
# Top-level Forecaster
# ---------------------------------------------------------------------------

class Forecaster:
    """
    Public interface for the probabilistic forecasting pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Combined daily DataFrame from nasa_loader.py. Must contain columns:
        date, location, solar_irradiance_kwh_m2_day, wind_speed_hub_m_s.
    variable : str
        "solar" or "wind".
    model_dir : Optional[Path]
        Directory for saving/loading trained LSTM weights.
        Defaults to outputs/models/.
    """

    COL_MAP = {
        "solar": "solar_irradiance_kwh_m2_day",
        "wind": "wind_speed_hub_m_s",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        variable: str = "solar",
        model_dir: Optional[Path] = None,
    ) -> None:
        if variable not in self.COL_MAP:
            raise ValueError(f"variable must be one of {list(self.COL_MAP)}")
        self.variable = variable
        self._col = self.COL_MAP[variable]
        self._df = df.copy()
        self._df["date"] = pd.to_datetime(self._df["date"])

        self._model_dir = model_dir or Path("outputs/models")
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._decomposers: Dict[str, SignalDecomposer] = {}
        self._emp_forecasters: Dict[str, EmpiricalQuantileForecaster] = {}
        self._persistence_forecasters: Dict[str, PersistenceForecaster] = {}
        self._lstm_models: Dict[str, _QuantileLSTMModel] = {}

        self._empirical_fitted: bool = False
        self._lstm_fitted_sites: List[str] = []

    def _get_site_series(self, site: str) -> pd.Series:
        mask = self._df["location"] == site
        sub = (
            self._df.loc[mask, ["date", self._col]]
            .dropna()
            .sort_values("date")
            .drop_duplicates("date")
            .set_index("date")
        )
        if sub.empty:
            raise ValueError(
                f"No data found for site='{site}', "
                f"variable='{self.variable}'."
            )
        return sub[self._col].asfreq("D").interpolate("linear")

    def fit_empirical(
        self, sites: Optional[List[str]] = None
    ) -> "Forecaster":
        available = self._df["location"].unique().tolist()
        sites = sites or available

        for site in sites:
            if site not in available:
                logger.warning("Site '%s' not found — skipping.", site)
                continue
            series = self._get_site_series(site)
            dc = SignalDecomposer().fit(series)
            self._decomposers[site] = dc
            self._emp_forecasters[site] = EmpiricalQuantileForecaster(dc).fit()
            self._persistence_forecasters[site] = PersistenceForecaster(dc)
            logger.info(
                "Empirical forecaster fitted for %s (%d days).",
                site, len(series),
            )

        self._empirical_fitted = True
        return self

    def tune_lstm(
        self,
        site: str,
        n_trials: int = 20,
        val_frac: float = 0.15,
        epochs_per_trial: int = 30,
        patience_per_trial: int = 5,
        verbose: bool = True,
        force_retune: bool = False,
    ) -> Dict:
        """
        Run Optuna search for one site. Must call fit_empirical() first.
        Returns best_params dict.
        Call fit_lstm() immediately after to train with best params.
        """
        if not self._empirical_fitted:
            raise RuntimeError("Call fit_empirical() before tune_lstm().")

        params_path = (
            self._model_dir
            / f"optuna_{self.variable}_{site.replace(' ', '_')}.npy"
        )

        model = _QuantileLSTMModel()

        if params_path.exists() and not force_retune:
            best_params = dict(np.load(params_path, allow_pickle=True).item())
            logger.info(
                "Loaded existing Optuna params for %s: %s", site, best_params
            )
            model.best_params = best_params
            model.input_window = best_params["input_window"]
            model.hidden_size = best_params["hidden_size"]
            model.num_layers = best_params["num_layers"]
            model.dropout = best_params["dropout"]
            self._lstm_models[site] = model
            return best_params

        residuals = self._decomposers[site].residuals.values
        best_params = model.tune(
            residuals,
            n_trials=n_trials,
            val_frac=val_frac,
            epochs_per_trial=epochs_per_trial,
            patience_per_trial=patience_per_trial,
            verbose=verbose,
        )
        np.save(params_path, best_params)
        self._lstm_models[site] = model
        return best_params

    def fit_lstm(
        self,
        site: str,
        epochs: int = 100,
        val_frac: float = 0.15,
        patience: int = 10,
        gap_threshold: float = GAP_THRESHOLD,
        verbose: bool = True,
        force_retrain: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """
        Train LSTM for one site.
        If tune_lstm() was called first, uses Optuna best params.
        Otherwise uses defaults.

        Returns (train_losses, val_losses) per epoch.
        """
        if not self._empirical_fitted:
            raise RuntimeError("Call fit_empirical() before fit_lstm().")

        model_path = (
            self._model_dir
            / f"lstm_{self.variable}_{site.replace(' ', '_')}.pt"
        )

        # Reuse existing model object from tune_lstm if available
        if site not in self._lstm_models:
            self._lstm_models[site] = _QuantileLSTMModel()

        model = self._lstm_models[site]

        if model_path.exists() and not force_retrain:
            logger.info("Loading existing LSTM from %s", model_path)
            model.load(model_path)
            if site not in self._lstm_fitted_sites:
                self._lstm_fitted_sites.append(site)
            return model.train_losses, model.val_losses

        residuals = self._decomposers[site].residuals.values
        if len(residuals) < MIN_TRAIN_DAYS:
            raise ValueError(
                f"Site '{site}' has only {len(residuals)} days; "
                f"need >= {MIN_TRAIN_DAYS} to train LSTM."
            )

        train_losses, val_losses = model.fit(
            residuals,
            epochs=epochs,
            val_frac=val_frac,
            patience=patience,
            gap_threshold=gap_threshold,
            verbose=verbose,
        )
        model.save(model_path)
        if site not in self._lstm_fitted_sites:
            self._lstm_fitted_sites.append(site)
        return train_losses, val_losses

    def compute_shap(
        self,
        site: str,
        n_background: int = 100,
        n_explain: int = 50,
    ) -> np.ndarray:
        if site not in self._lstm_models:
            raise RuntimeError(f"Train LSTM for '{site}' first.")
        return self._lstm_models[site].compute_shap(n_background, n_explain)

    def explain_forecast_lime(
        self,
        site: str,
        anchor_date: Optional[pd.Timestamp] = None,
        quantile_idx: int = 2,
        horizon_idx: int = 0,
        num_features: int = 10,
    ) -> object:
        if site not in self._lstm_models:
            raise RuntimeError(f"Train LSTM for '{site}' first.")
        series = self._get_site_series(site)
        if anchor_date is None:
            anchor_date = series.index[-1]
        resid = self._decomposers[site].residuals
        context = resid[resid.index <= anchor_date].values[-self._lstm_models[site].input_window:]
        iw = self._lstm_models[site].input_window
        if len(context) < iw:
            pad = np.zeros(iw - len(context))
            context = np.concatenate([pad, context])
        return self._lstm_models[site].explain_forecast_lime(
            context, quantile_idx, horizon_idx, num_features
        )

    def forecast_empirical(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if not self._empirical_fitted or site not in self._emp_forecasters:
            raise RuntimeError(
                f"fit_empirical() not yet run for site '{site}'."
            )
        if anchor_date is None:
            anchor_date = self._get_site_series(site).index[-1]
        fan = self._emp_forecasters[site].forecast(anchor_date, n_days=n_days)
        fan.attrs.update(
            {"site": site, "variable": self.variable, "method": "empirical"}
        )
        return fan

    def forecast_persistence(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if site not in self._persistence_forecasters:
            raise RuntimeError(
                f"fit_empirical() not yet run for site '{site}'."
            )
        if anchor_date is None:
            anchor_date = self._get_site_series(site).index[-1]
        fan = self._persistence_forecasters[site].forecast(
            anchor_date, n_days=n_days
        )
        fan.attrs.update(
            {"site": site, "variable": self.variable, "method": "persistence"}
        )
        return fan

    def forecast_lstm(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if site not in self._lstm_models or not self._lstm_models[site]._is_trained:
            raise RuntimeError(
                f"LSTM not trained for '{site}'. "
                f"Call fit_lstm('{site}') first."
            )

        dc = self._decomposers[site]
        model = self._lstm_models[site]
        series = self._get_site_series(site)

        if anchor_date is None:
            anchor_date = series.index[-1]

        resid = dc.residuals
        iw = model.input_window
        context = resid[resid.index <= anchor_date].values[-iw:]
        if len(context) < iw:
            pad = np.zeros(iw - len(context))
            context = np.concatenate([pad, context])

        lstm_resid = model.predict(context)

        future_dates = pd.date_range(
            start=anchor_date + pd.Timedelta(days=1), periods=n_days, freq="D"
        )
        if n_days < FORECAST_HORIZON:
            lstm_resid = lstm_resid[:, :n_days]
        elif n_days > FORECAST_HORIZON:
            pad = np.repeat(
                lstm_resid[:, -1:], n_days - FORECAST_HORIZON, axis=1
            )
            lstm_resid = np.concatenate([lstm_resid, pad], axis=1)

        trend = dc.trend_at(future_dates)
        seasonal = dc.seasonal_at(future_dates)
        base = trend + seasonal

        rows = []
        for i, dt in enumerate(future_dates):
            row = {"date": dt}
            for qi, lbl in enumerate(QUANTILE_LABELS):
                row[lbl] = base[i] + float(lstm_resid[qi, i])
            rows.append(row)

        fan = pd.DataFrame(rows).set_index("date").clip(lower=0.0)
        fan.attrs.update(
            {"site": site, "variable": self.variable, "method": "lstm"}
        )
        return fan

    def forecast(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Auto-select: LSTM if trained, else empirical."""
        if (
            site in self._lstm_models
            and self._lstm_models[site]._is_trained
        ):
            return self.forecast_lstm(
                site, n_days=n_days, anchor_date=anchor_date
            )
        return self.forecast_empirical(
            site, n_days=n_days, anchor_date=anchor_date
        )

    def evaluate(
        self,
        site: str,
        method: str = "empirical",
        test_days: int = 90,
    ) -> pd.DataFrame:
        """
        Walk-forward pinball loss on last test_days of data.
        method : "empirical" | "lstm" | "persistence"
        """
        series = self._get_site_series(site)
        n = len(series)
        iw = (
            self._lstm_models[site].input_window
            if site in self._lstm_models
            else DEFAULT_INPUT_WINDOW
        )
        if n < iw + test_days + 1:
            raise ValueError("Not enough data for evaluation.")

        cut_idx = n - test_days
        records = []

        for step in range(test_days):
            anchor = series.index[cut_idx + step - 1]
            true_val = float(series.iloc[cut_idx + step])

            if method == "lstm" and site in self._lstm_models:
                fan = self.forecast_lstm(site, n_days=1, anchor_date=anchor)
            elif method == "persistence":
                fan = self.forecast_persistence(
                    site, n_days=1, anchor_date=anchor
                )
            else:
                fan = self.forecast_empirical(
                    site, n_days=1, anchor_date=anchor
                )

            for q, lbl in zip(QUANTILES, QUANTILE_LABELS):
                pred = float(fan.iloc[0][lbl])
                records.append(
                    {
                        "date": fan.index[0],
                        "quantile": q,
                        "label": lbl,
                        "y_true": true_val,
                        "y_pred": pred,
                        "pinball_loss": pinball_loss_numpy(
                            np.array([true_val]), np.array([pred]), q
                        ),
                    }
                )

        return pd.DataFrame(records)

    def compare_methods(
        self,
        site: str,
        test_days: int = 90,
    ) -> pd.DataFrame:
        """
        Mean pinball loss per quantile per method.
        Produces the results table for the IEEE paper.
        """
        methods = ["persistence", "empirical"]
        if site in self._lstm_models and self._lstm_models[site]._is_trained:
            methods.append("lstm")

        rows = []
        for method in methods:
            eval_df = self.evaluate(
                site, method=method, test_days=test_days
            )
            for lbl, grp in eval_df.groupby("label"):
                rows.append(
                    {
                        "method": method,
                        "quantile": lbl,
                        "mean_pinball": round(
                            grp["pinball_loss"].mean(), 6
                        ),
                    }
                )

        return (
            pd.DataFrame(rows)
            .pivot(
                index="quantile", columns="method", values="mean_pinball"
            )
            .reset_index()
        )

    @staticmethod
    def compute_quantiles(
        df: pd.DataFrame,
        col: str,
        quantiles: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        quantiles = quantiles or QUANTILES
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["doy"] = df["date"].dt.dayofyear

        results = []
        for (loc, doy), grp in df.groupby(["location", "doy"]):
            vals = grp[col].dropna().values
            if len(vals) == 0:
                continue
            row: Dict = {"location": loc, "doy": int(doy)}
            for q, lbl in zip(quantiles, QUANTILE_LABELS):
                row[lbl] = float(np.quantile(vals, q))
            results.append(row)

        return pd.DataFrame(results).set_index(["location", "doy"])


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def build_forecasters(
    df: pd.DataFrame,
    model_dir: Optional[Path] = None,
    fit_empirical: bool = True,
) -> Dict[str, "Forecaster"]:
    """
    Build and optionally fit empirical forecasters for solar and wind.
    Returns dict with keys "solar" and "wind".
    """
    forecasters: Dict[str, Forecaster] = {}
    for var in VARIABLES:
        f = Forecaster(df, variable=var, model_dir=model_dir)
        if fit_empirical:
            f.fit_empirical()
        forecasters[var] = f
    return forecasters


# ---------------------------------------------------------------------------
# ForecastBundle
# ---------------------------------------------------------------------------

class ForecastBundle:
    """
    Joint operational forecast for one site with solar-wind correlation.
    """

    def __init__(
        self,
        solar_forecaster: Forecaster,
        wind_forecaster: Forecaster,
        correlation_estimator: SiteCorrelationEstimator,
    ) -> None:
        self._sf = solar_forecaster
        self._wf = wind_forecaster
        self._corr = correlation_estimator

    def forecast(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        sol_fan = self._sf.forecast(
            site, n_days=n_days, anchor_date=anchor_date
        )
        wnd_fan = self._wf.forecast(
            site, n_days=n_days, anchor_date=anchor_date
        )
        sol_fan.columns = [f"solar_{c}" for c in sol_fan.columns]
        wnd_fan.columns = [f"wind_{c}" for c in wnd_fan.columns]
        bundle = sol_fan.join(wnd_fan)
        bundle["rho"] = [
            self._corr.get(site, dt.month) for dt in bundle.index
        ]
        bundle.attrs.update({"site": site, "n_days": n_days})
        return bundle
