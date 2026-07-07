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

# Suppress statsmodels convergence warnings globally
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*converge.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

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
        trend = raw_trend.ffill().bfill()
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
        smooth = pd.Series(profile[1:]).rolling(15, center=True, min_periods=1).mean()
        self._seasonal_profile = np.concatenate([[0.0], smooth.values])

        seasonal_vals = np.array([self._seasonal_profile[d] for d in doy])
        residual = detrended - seasonal_vals
        self._residual = pd.Series(residual.values, index=series.index, name="residual")

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
            self._doy_quantiles[doy] = {q: float(np.quantile(vals, q)) for q in QUANTILES}
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
            row = {lbl: base + q_dict[q] for q, lbl in zip(QUANTILES, QUANTILE_LABELS)}
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
            raise ImportError("PyTorch is required. Install with: pip install torch")
        import torch

        self.input_window = input_window
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_quantiles = len(QUANTILES)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._net = _build_net(hidden_size, num_layers, dropout, self.n_quantiles, horizon).to(
            self._device
        )

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
            p = preds[:, qi, :]  # (B, H)
            t = targets[:, qi, :] if targets.dim() == 3 else targets  # (B, H)
            err = t - p
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
        epochs_per_trial: int = 15,
        patience_per_trial: int = 3,
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

        Returns best_params dict (keys: hidden_size, num_layers, dropout,
        lr, batch_size, input_window) and rebuilds self._net with the
        best configuration so fit() can be called immediately after.
        """
        try:
            import optuna
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("Install with: pip install optuna torch")

        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING
        )

        scaler_mean = float(np.mean(residuals))
        scaler_std = float(np.std(residuals)) or 1.0
        scaled = (residuals - scaler_mean) / scaler_std

        def objective(trial: "optuna.Trial") -> float:
            hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 96, 128])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            input_window = trial.suggest_categorical("input_window", [30, 45, 60, 90])

            net = _build_net(
                hidden_size, num_layers, dropout, self.n_quantiles, self.horizon
            ).to(self._device)

            X, Y = self._make_sequences(scaled, input_window)
            if len(X) < 20:
                raise optuna.exceptions.TrialPruned()
            n_val = max(1, int(len(X) * val_frac))
            X_tr, Y_tr = X[:-n_val], Y[:-n_val]
            X_val, Y_val = X[-n_val:], Y[-n_val:]

            X_tr_t = torch.tensor(X_tr[:, :, None]).to(self._device)
            Y_tr_t = torch.tensor(Y_tr).to(self._device)
            X_val_t = torch.tensor(X_val[:, :, None]).to(self._device)
            Y_val_t = torch.tensor(Y_val).to(self._device)

            ds = TensorDataset(X_tr_t, Y_tr_t)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            optim = torch.optim.Adam(net.parameters(), lr=lr)

            best_val = math.inf
            patience_counter = 0

            for epoch in range(1, epochs_per_trial + 1):
                net.train()
                for xb, yb in loader:
                    optim.zero_grad()
                    preds = net(xb)
                    loss = self._pinball_loss_torch(preds, yb.unsqueeze(1).expand_as(preds))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optim.step()

                net.eval()
                with torch.no_grad():
                    val_preds = net(X_val_t)
                    val_loss = self._pinball_loss_torch(
                        val_preds, Y_val_t.unsqueeze(1).expand_as(val_preds)
                    )
                vl = float(val_loss.item())

                if vl < best_val:
                    best_val = vl
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience_per_trial:
                        break

                trial.report(vl, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return best_val

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(objective, n_trials=n_trials)

        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not completed:
            raise RuntimeError(
                "LSTM Optuna study: all trials failed or were pruned — "
                "check the warnings above before proceeding."
            )

        best = dict(study.best_params)
        self.best_params = best
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

        logger.info("Optuna best params: %s", best)
        logger.info("Optuna best val pinball: %.5f", study.best_value)
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
                loss = self._pinball_loss_torch(preds, yb.unsqueeze(1).expand_as(preds))
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
                    epoch,
                    epochs,
                    train_loss,
                    vl,
                    gap,
                )

            if vl < best_val:
                best_val = vl
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stop (patience=%d) at epoch %d",
                        patience,
                        epoch,
                    )
                    break

            if gap > gap_threshold:
                logger.info(
                    "Early stop (gap=%.4f > %.4f) at epoch %d",
                    gap,
                    gap_threshold,
                    epoch,
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
        x = torch.tensor(scaled[None, :, None], dtype=torch.float32).to(self._device)
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
        background = torch.tensor(self._X_train[idx_bg, :, None], dtype=torch.float32).to(
            self._device
        )

        n_exp = min(n_explain, len(self._X_val))
        X_exp = torch.tensor(self._X_val[-n_exp:, :, None], dtype=torch.float32).to(
            self._device
        )

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

        # cuDNN's optimized LSTM kernel refuses to run backward() while the
        # module is in eval mode (it discards the buffers backward needs, as
        # an inference-only speed optimization). shap.GradientExplainer calls
        # backward() internally, so disable cuDNN for just this call — this
        # falls back to PyTorch's native LSTM path, which supports backward
        # in eval mode. Dropout/eval behavior is unaffected; only the CUDA
        # kernel used changes.
        cudnn_was_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            explainer = shap.GradientExplainer(flat_net, background)
            shap_vals = explainer.shap_values(X_exp)
        finally:
            torch.backends.cudnn.enabled = cudnn_was_enabled

        stacked = np.stack([sv[:, :, 0] for sv in shap_vals], axis=-1)
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
            X_3d = X_flat.reshape(-1, self.input_window, 1).astype(np.float32)
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

        scaled_input = (last_residuals - self._scaler_mean) / self._scaler_std
        explanation = explainer.explain_instance(
            scaled_input,
            _predict_fn,
            num_features=num_features,
        )
        logger.info("LIME top feature: %s", explanation.as_list()[0][0])
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

    def load(self, path: Path, residuals: Optional[np.ndarray] = None) -> None:
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
        if residuals is not None:
            scaled = (residuals - self._scaler_mean) / self._scaler_std
            X, _ = self._make_sequences(scaled, self.input_window)
            n_val = max(1, int(len(X) * 0.15))
            self._X_train = X[:-n_val].copy()
            self._X_val = X[-n_val:].copy()
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
# Phase 3 — N-BEATS (neuralforecast), same rigor as the LSTM path
# ---------------------------------------------------------------------------


def _try_import_neuralforecast() -> bool:
    try:
        import neuralforecast  # noqa: F401

        return True
    except ImportError:
        return False


def _new_nbeats_loss_history():
    """
    Return a fresh PyTorch Lightning Callback instance that records real
    per-epoch train/val loss, since Lightning's `callback_metrics` only
    holds the latest scalar value, not a full history. Import is deferred
    so forecaster.py can still be imported when pytorch_lightning (an
    N-BEATS-only dependency) isn't installed.
    """
    from pytorch_lightning.callbacks import Callback

    class _LossHistory(Callback):
        def __init__(self):
            self.train_losses: List[float] = []
            self.val_losses: List[float] = []

        def on_train_epoch_end(self, trainer, pl_module):
            for key in ("train_loss_epoch", "train_loss"):
                if key in trainer.callback_metrics:
                    self.train_losses.append(float(trainer.callback_metrics[key]))
                    break

        def on_validation_epoch_end(self, trainer, pl_module):
            for key in ("valid_loss", "ptl/val_loss"):
                if key in trainer.callback_metrics:
                    self.val_losses.append(float(trainer.callback_metrics[key]))
                    break

    return _LossHistory()


class _NBEATSModel:
    """
    N-BEATS quantile forecaster (via neuralforecast) with Optuna
    hyperparameter search, patience-based early stopping, and
    model-agnostic SHAP/LIME explanations on the input residual window.
    """

    def __init__(
        self,
        input_window: int = DEFAULT_INPUT_WINDOW,
        horizon: int = FORECAST_HORIZON,
        max_steps: int = 500,
    ) -> None:
        if not _try_import_neuralforecast():
            raise ImportError(
                "neuralforecast is required. Install with: pip install neuralforecast"
            )
        self.input_window = input_window
        self.horizon = horizon
        self.max_steps = max_steps
        self.best_params: Dict = {}
        self._nf = None
        self._site: Optional[str] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self._is_trained: bool = False

    def _build_model(
        self,
        input_size,
        learning_rate,
        n_blocks,
        mlp_width,
        patience=None,
        max_steps=None,
        callbacks=None,
    ):
        from neuralforecast.models import NBEATS
        from neuralforecast.losses.pytorch import MQLoss
        import torch

        kwargs = dict(
            h=self.horizon,
            input_size=input_size,
            loss=MQLoss(quantiles=QUANTILES),
            learning_rate=learning_rate,
            n_blocks=[n_blocks, n_blocks, n_blocks],
            mlp_units=[[mlp_width, mlp_width]] * 3,
            max_steps=max_steps or self.max_steps,
            scaler_type="standard",
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )
        # early_stop_patience_steps is set EXPLICITLY in both branches below
        # (never omitted) so behavior does not depend on whatever default
        # value a given neuralforecast version ships with. -1 disables
        # early stopping outright; this is what we want during Optuna/CV
        # trials, since Lightning's EarlyStopping cannot reliably see
        # `ptl/val_loss` inside per-window cross_validation() refits.
        if patience is not None:
            kwargs["early_stop_patience_steps"] = patience
            kwargs["val_check_steps"] = 10
        else:
            kwargs["early_stop_patience_steps"] = -1
        if callbacks:
            kwargs["callbacks"] = callbacks
        return NBEATS(**kwargs)

    def _to_nf_frame(self, residuals: np.ndarray, site: str) -> pd.DataFrame:
        dates = pd.date_range("2000-01-01", periods=len(residuals), freq="D")
        return pd.DataFrame({"unique_id": site, "ds": dates, "y": residuals.astype(float)})

    # ------------------------------------------------------------------
    def tune(
        self,
        residuals: np.ndarray,
        site: str,
        n_trials: int = 20,
        n_windows: int = 3,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Optuna search over N-BEATS hyperparameters (input_size multiple,
        learning_rate, stack depth, MLP width), scored on walk-forward
        cross-validation. Patience is enforced inside each trial via
        early_stop_patience_steps, same principle as the LSTM's
        patience_per_trial.
        """
        try:
            import optuna
            from neuralforecast import NeuralForecast
        except ImportError:
            raise ImportError("Install with: pip install optuna neuralforecast")

        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING
        )
        nf_df = self._to_nf_frame(residuals, site)

        def objective(trial: "optuna.Trial") -> float:
            mult = trial.suggest_categorical("input_window_mult", [2, 3, 4, 6])
            lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            n_blocks = trial.suggest_int("n_blocks", 1, 3)
            mlp_width = trial.suggest_categorical("mlp_units_width", [128, 256, 512])

            model = self._build_model(
                mult * self.horizon, lr, n_blocks, mlp_width, patience=None
            )
            logger.info(
                "Trial model early_stop_patience_steps=%s (expect -1)",
                getattr(model, "early_stop_patience_steps", "MISSING_ATTR"),
            )
            nf = NeuralForecast(models=[model], freq="D")
            try:
                cv_df = nf.cross_validation(
                    nf_df,
                    n_windows=n_windows,
                    step_size=self.horizon,
                    val_size=self.horizon,
                )
            except Exception as exc:
                logger.warning("N-BEATS trial failed: %s", exc)
                raise optuna.exceptions.TrialPruned()

            q_cols = [c for c in cv_df.columns if c not in ("unique_id", "ds", "cutoff", "y")]
            if not q_cols:
                raise optuna.exceptions.TrialPruned()
            q50_col = q_cols[len(QUANTILES) // 2]
            return pinball_loss_numpy(
                cv_df["y"].values, np.clip(cv_df[q50_col].values, 0, None), 0.50
            )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(objective, n_trials=n_trials)

        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not completed:
            raise RuntimeError(
                "N-BEATS Optuna study: all trials failed or were pruned — "
                "check the warnings above before proceeding."
            )

        best = dict(study.best_params)
        best["input_size"] = best.pop("input_window_mult") * self.horizon
        self.best_params = best
        self.input_window = best["input_size"]
        logger.info("N-BEATS Optuna best params: %s", best)
        logger.info("N-BEATS Optuna best val pinball (Q50): %.5f", study.best_value)
        return best

    # ------------------------------------------------------------------
    def fit(
        self, residuals: np.ndarray, site: str, patience: int = 10, verbose: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Train final N-BEATS model with Optuna best params (or defaults),
        with patience-based early stopping via early_stop_patience_steps.
        Captures real per-epoch train/val loss via a Lightning callback
        (self.train_losses / self.val_losses), same return shape as the
        LSTM's fit().
        """
        from neuralforecast import NeuralForecast

        p = self.best_params or {
            "input_size": self.input_window,
            "learning_rate": 1e-3,
            "n_blocks": 2,
            "mlp_units_width": 256,
        }

        history = _new_nbeats_loss_history()

        def _build(patience_val):
            return self._build_model(
                p["input_size"],
                p["learning_rate"],
                p["n_blocks"],
                p["mlp_units_width"],
                patience=patience_val,
                callbacks=[history],
            )

        nf_df = self._to_nf_frame(residuals, site)

        try:
            model = _build(patience)
            self._nf = NeuralForecast(models=[model], freq="D")
            self._nf.fit(nf_df, val_size=self.horizon * 3)
        except Exception as e:
            logger.warning(
                "N-BEATS fit with early stopping failed (%s); retrying without patience.", e
            )
            history = _new_nbeats_loss_history()
            model = self._build_model(
                p["input_size"],
                p["learning_rate"],
                p["n_blocks"],
                p["mlp_units_width"],
                patience=None,
                callbacks=[history],
            )
            self._nf = NeuralForecast(models=[model], freq="D")
            self._nf.fit(nf_df, val_size=self.horizon * 3)

        self.train_losses = history.train_losses
        self.val_losses = history.val_losses
        self._site = site
        self._is_trained = True
        return self.train_losses, self.val_losses

    # ------------------------------------------------------------------
    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Model-agnostic wrapper for SHAP/LIME: X is (n_samples, input_window)."""
        preds = []
        for row in X:
            df = self._to_nf_frame(row, self._site)
            fcst = self._nf.predict(df=df)
            q_cols = [c for c in fcst.columns if c not in ("unique_id", "ds")]
            preds.append(float(fcst[q_cols[len(QUANTILES) // 2]].iloc[0]))
        return np.array(preds)

    def compute_shap(
        self, residual_windows: np.ndarray, n_background: int = 30, n_explain: int = 20
    ) -> np.ndarray:
        """
        Model-agnostic SHAP via KernelExplainer, since NeuralForecast
        doesn't expose a plain forward() like the hand-rolled LSTM.
        NOTE: test this against your installed neuralforecast version —
        nf.predict(df=...) semantics can vary slightly across versions.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Install with: pip install shap")
        if not self._is_trained:
            raise RuntimeError("Call fit() before compute_shap().")

        background = residual_windows[:n_background]
        X_exp = residual_windows[n_background : n_background + n_explain]
        explainer = shap.KernelExplainer(self._predict_fn, background)
        shap_vals = explainer.shap_values(X_exp, nsamples=100)
        importance = np.mean(np.abs(shap_vals), axis=0)
        self.shap_importance = importance
        logger.info("N-BEATS SHAP top-3 lags: %s", np.argsort(importance)[::-1][:3].tolist())
        return importance

    def explain_lime(
        self, residual_windows: np.ndarray, instance: np.ndarray, num_features: int = 10
    ):
        try:
            from lime import lime_tabular
        except ImportError:
            raise ImportError("Install with: pip install lime")
        if not self._is_trained:
            raise RuntimeError("Call fit() before explain_lime().")

        explainer = lime_tabular.LimeTabularExplainer(
            residual_windows,
            mode="regression",
            feature_names=[f"lag_{i}" for i in range(residual_windows.shape[1])],
        )
        explanation = explainer.explain_instance(
            instance, self._predict_fn, num_features=num_features
        )
        logger.info("N-BEATS LIME top feature: %s", explanation.as_list()[0][0])
        return explanation


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

        self._model_dir = Path(model_dir) if model_dir is not None else Path("outputs/models")
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._decomposers: Dict[str, SignalDecomposer] = {}
        self._emp_forecasters: Dict[str, EmpiricalQuantileForecaster] = {}
        self._persistence_forecasters: Dict[str, PersistenceForecaster] = {}
        self._ets_forecasters: Dict[str, ETSForecaster] = {}
        self._stlets_forecasters: Dict[str, STLETSForecaster] = {}
        self._lstm_models: Dict[str, _QuantileLSTMModel] = {}
        self._nbeats_models: Dict[str, "_NBEATSModel"] = {}

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
                f"No data found for site='{site}', " f"variable='{self.variable}'."
            )
        return sub[self._col].asfreq("D").interpolate("linear")

    def fit_empirical(self, sites: Optional[List[str]] = None) -> "Forecaster":
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
            self._ets_forecasters[site] = ETSForecaster(dc).fit(series)
            self._stlets_forecasters[site] = STLETSForecaster(dc).fit(series)
            logger.info(
                "Empirical forecaster fitted for %s (%d days).",
                site,
                len(series),
            )

        self._empirical_fitted = True
        return self

    def tune_lstm(
        self,
        site: str,
        n_trials: int = 20,
        val_frac: float = 0.15,
        epochs_per_trial: int = 15,
        patience_per_trial: int = 3,
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

        params_path = self._model_dir / f"optuna_{self.variable}_{site.replace(' ', '_')}.npy"

        model = _QuantileLSTMModel()

        if params_path.exists() and not force_retune:
            best_params = dict(np.load(params_path, allow_pickle=True).item())
            logger.info("Loaded existing Optuna params for %s: %s", site, best_params)
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

        model_path = self._model_dir / f"lstm_{self.variable}_{site.replace(' ', '_')}.pt"

        # Reuse existing model object from tune_lstm if available
        if site not in self._lstm_models:
            self._lstm_models[site] = _QuantileLSTMModel()

        model = self._lstm_models[site]

        if model_path.exists() and not force_retrain:
            logger.info("Loading existing LSTM from %s", model_path)
            residuals = self._decomposers[site].residuals.values
            model.load(model_path, residuals=residuals)
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
        context = resid[resid.index <= anchor_date].values[
            -self._lstm_models[site].input_window :
        ]
        iw = self._lstm_models[site].input_window
        if len(context) < iw:
            pad = np.zeros(iw - len(context))
            context = np.concatenate([pad, context])
        return self._lstm_models[site].explain_forecast_lime(
            context, quantile_idx, horizon_idx, num_features
        )

    # ------------------------------------------------------------------
    def tune_nbeats(
        self,
        site: str,
        n_trials: int = 20,
        patience: int = 10,
        verbose: bool = True,
        force_retune: bool = False,
    ) -> Dict:
        """Run Optuna search for N-BEATS at one site. Mirrors tune_lstm()."""
        if not self._empirical_fitted:
            raise RuntimeError("Call fit_empirical() before tune_nbeats().")

        params_path = (
            self._model_dir / f"optuna_nbeats_{self.variable}_{site.replace(' ', '_')}.npy"
        )
        model = _NBEATSModel()

        if params_path.exists() and not force_retune:
            best_params = dict(np.load(params_path, allow_pickle=True).item())
            model.best_params = best_params
            model.input_window = best_params["input_size"]
            self._nbeats_models[site] = model
            return best_params

        residuals = self._decomposers[site].residuals.values
        best_params = model.tune(
            residuals, site, n_trials=n_trials, patience=patience, verbose=verbose
        )
        np.save(params_path, best_params)
        self._nbeats_models[site] = model
        return best_params

    def fit_nbeats(
        self, site: str, patience: int = 10, verbose: bool = True
    ) -> Tuple[List[float], List[float]]:
        """Train final N-BEATS model for one site. Mirrors fit_lstm()."""
        if site not in self._nbeats_models:
            self._nbeats_models[site] = _NBEATSModel()
        residuals = self._decomposers[site].residuals.values
        return self._nbeats_models[site].fit(
            residuals, site, patience=patience, verbose=verbose
        )

    def compute_shap_nbeats(
        self, site: str, n_background: int = 30, n_explain: int = 20
    ) -> np.ndarray:
        residuals = self._decomposers[site].residuals.values
        iw = self._nbeats_models[site].input_window
        windows = np.array([residuals[i : i + iw] for i in range(len(residuals) - iw)])
        return self._nbeats_models[site].compute_shap(windows, n_background, n_explain)

    def forecast_nbeats(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """N-BEATS forecast, same interface as forecast_lstm()."""
        if site not in self._nbeats_models or not self._nbeats_models[site]._is_trained:
            raise RuntimeError(
                f"N-BEATS not trained for '{site}'. Call fit_nbeats('{site}') first."
            )

        dc = self._decomposers[site]
        model = self._nbeats_models[site]
        series = self._get_site_series(site)

        if anchor_date is None:
            anchor_date = series.index[-1]

        resid = dc.residuals
        iw = model.input_window
        context = resid[resid.index <= anchor_date].values[-iw:]
        if len(context) < iw:
            pad = np.zeros(iw - len(context))
            context = np.concatenate([pad, context])

        context_df = model._to_nf_frame(context, site)
        fcst = model._nf.predict(df=context_df)
        q_cols = [c for c in fcst.columns if c not in ("unique_id", "ds")]

        future_dates = pd.date_range(
            start=anchor_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq="D"
        )
        rows = []
        for i, dt in enumerate(future_dates[:n_days]):
            row = {"date": dt}
            for qi, lbl in enumerate(QUANTILE_LABELS):
                col = q_cols[qi] if qi < len(q_cols) else q_cols[-1]
                row[lbl] = (
                    float(fcst[col].iloc[i]) if i < len(fcst) else float(fcst[col].iloc[-1])
                )
            rows.append(row)

        fan = pd.DataFrame(rows).set_index("date").clip(lower=0.0)
        fan.attrs.update({"site": site, "variable": self.variable, "method": "nbeats"})
        return fan

    # ------------------------------------------------------------------

    def forecast_empirical(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if not self._empirical_fitted or site not in self._emp_forecasters:
            raise RuntimeError(f"fit_empirical() not yet run for site '{site}'.")
        if anchor_date is None:
            anchor_date = self._get_site_series(site).index[-1]
        fan = self._emp_forecasters[site].forecast(anchor_date, n_days=n_days)
        fan.attrs.update({"site": site, "variable": self.variable, "method": "empirical"})
        return fan

    def forecast_persistence(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if site not in self._persistence_forecasters:
            raise RuntimeError(f"fit_empirical() not yet run for site '{site}'.")
        if anchor_date is None:
            anchor_date = self._get_site_series(site).index[-1]
        fan = self._persistence_forecasters[site].forecast(anchor_date, n_days=n_days)
        fan.attrs.update({"site": site, "variable": self.variable, "method": "persistence"})
        return fan

    def forecast_ets(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if site not in self._ets_forecasters:
            raise RuntimeError(f"fit_empirical() not yet run for site '{site}'.")
        if anchor_date is None:
            anchor_date = self._get_site_series(site).index[-1]
        fan = self._ets_forecasters[site].forecast(anchor_date, n_days=n_days)
        fan.attrs.update({"site": site, "variable": self.variable, "method": "ets"})
        return fan

    def forecast_stl_ets(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if site not in self._stlets_forecasters:
            raise RuntimeError(f"fit_empirical() not yet run for site '{site}'.")
        if anchor_date is None:
            anchor_date = self._get_site_series(site).index[-1]
        fan = self._stlets_forecasters[site].forecast(anchor_date, n_days=n_days)
        fan.attrs.update({"site": site, "variable": self.variable, "method": "stl_ets"})
        return fan

    def forecast_lstm(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if site not in self._lstm_models or not self._lstm_models[site]._is_trained:
            raise RuntimeError(
                f"LSTM not trained for '{site}'. " f"Call fit_lstm('{site}') first."
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
            pad = np.repeat(lstm_resid[:, -1:], n_days - FORECAST_HORIZON, axis=1)
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
        fan.attrs.update({"site": site, "variable": self.variable, "method": "lstm"})
        return fan

    def forecast(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Auto-select: LSTM if trained, else empirical."""
        if site in self._lstm_models and self._lstm_models[site]._is_trained:
            return self.forecast_lstm(site, n_days=n_days, anchor_date=anchor_date)
        return self.forecast_empirical(site, n_days=n_days, anchor_date=anchor_date)

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
            elif method == "nbeats" and site in self._nbeats_models:
                fan = self.forecast_nbeats(site, n_days=1, anchor_date=anchor)
            elif method == "persistence":
                fan = self.forecast_persistence(site, n_days=1, anchor_date=anchor)
            elif method == "ets":
                fan = self.forecast_ets(site, n_days=1, anchor_date=anchor)
            elif method == "stl_ets":
                fan = self.forecast_stl_ets(site, n_days=1, anchor_date=anchor)
            else:
                fan = self.forecast_empirical(site, n_days=1, anchor_date=anchor)

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
        methods = ["persistence", "ets", "stl_ets", "empirical"]
        if site in self._lstm_models and self._lstm_models[site]._is_trained:
            methods.append("lstm")
        if site in self._nbeats_models and self._nbeats_models[site]._is_trained:
            methods.append("nbeats")

        rows = []
        for method in methods:
            eval_df = self.evaluate(site, method=method, test_days=test_days)
            for lbl, grp in eval_df.groupby("label"):
                rows.append(
                    {
                        "method": method,
                        "quantile": lbl,
                        "mean_pinball": round(grp["pinball_loss"].mean(), 6),
                    }
                )

        return (
            pd.DataFrame(rows)
            .pivot(index="quantile", columns="method", values="mean_pinball")
            .reset_index()
        )

    def run_diebold_mariano(
        self,
        site: str,
        reference_method: str = "lstm",
        quantile: float = 0.50,
        test_days: int = 90,
    ) -> pd.DataFrame:
        """
        Run DM test comparing reference_method vs all other methods.
        Returns DataFrame with dm_statistic, p_value, significant columns.
        """
        methods = ["persistence", "ets", "stl_ets", "empirical"]
        if site in self._lstm_models and self._lstm_models[site]._is_trained:
            methods.append("lstm")
        if site in self._nbeats_models and self._nbeats_models[site]._is_trained:
            methods.append("nbeats")

        eval_results = {}
        for method in methods:
            try:
                eval_results[method] = self.evaluate(site, method=method, test_days=test_days)
            except Exception as e:
                logger.warning("DM eval failed for %s: %s", method, e)

        if reference_method not in eval_results:
            raise ValueError(f"reference_method '{reference_method}' evaluation failed.")

        dm_table = run_diebold_mariano(
            eval_results,
            reference_method=reference_method,
            quantile=quantile,
        )
        dm_table.insert(0, "site", site)
        return dm_table

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
        sol_fan = self._sf.forecast(site, n_days=n_days, anchor_date=anchor_date)
        wnd_fan = self._wf.forecast(site, n_days=n_days, anchor_date=anchor_date)
        sol_fan.columns = [f"solar_{c}" for c in sol_fan.columns]
        wnd_fan.columns = [f"wind_{c}" for c in wnd_fan.columns]
        bundle = sol_fan.join(wnd_fan)
        bundle["rho"] = [self._corr.get(site, dt.month) for dt in bundle.index]
        bundle.attrs.update({"site": site, "n_days": n_days})
        return bundle


# ---------------------------------------------------------------------------
# Classical baselines — ETS and STL+ETS
# ---------------------------------------------------------------------------


class ETSForecaster:
    """
    Exponential Smoothing (Holt-Winters) baseline.

    Fits on the raw series (not residuals) and produces a point forecast
    repeated across all quantiles — same approach as PersistenceForecaster
    but adaptive rather than naive.

    Uses additive trend + additive seasonal (period=365).
    Falls back to simple exponential smoothing if seasonal fit fails.
    """

    def __init__(self, decomposer: SignalDecomposer) -> None:
        self._dc = decomposer
        self._model = None
        self._series = None
        self._is_fitted: bool = False

    def fit(self, series: pd.Series) -> "ETSForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._series = series.copy()
        try:
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=365,
                initialization_method="estimated",
            )
            self._model = model.fit(optimized=True)
            self._method = "holt_winters"
        except Exception:
            # Fallback: simple exponential smoothing
            model = ExponentialSmoothing(
                series,
                trend=None,
                seasonal=None,
                initialization_method="estimated",
            )
            self._model = model.fit(optimized=True)
            self._method = "simple_exp"
        self._is_fitted = True
        return self

    def forecast(
        self,
        anchor_date: pd.Timestamp,
        n_days: int = FORECAST_HORIZON,
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first.")

        # Get point forecast from ETS
        series_to_anchor = self._series[self._series.index <= anchor_date]
        steps = n_days

        try:
            import warnings
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self._method == "holt_winters":
                    model = ExponentialSmoothing(
                        series_to_anchor,
                        trend="add",
                        seasonal=None,
                        initialization_method="estimated",
                    )
                else:
                    model = ExponentialSmoothing(
                        series_to_anchor,
                        trend=None,
                        seasonal=None,
                        initialization_method="estimated",
                    )
                fitted = model.fit(optimized=True)
                point_forecasts = fitted.forecast(steps).values
        except Exception:
            # Ultimate fallback: last value repeated
            point_forecasts = np.full(steps, float(series_to_anchor.iloc[-1]))

        point_forecasts = np.clip(point_forecasts, 0, None)

        future_dates = pd.date_range(
            start=anchor_date + pd.Timedelta(days=1), periods=n_days, freq="D"
        )

        rows = []
        for i, dt in enumerate(future_dates):
            val = float(point_forecasts[i])
            row = {lbl: val for lbl in QUANTILE_LABELS}
            row["date"] = dt
            rows.append(row)

        fan = pd.DataFrame(rows).set_index("date")
        fan.attrs.update({"method": "ets"})
        return fan


class STLETSForecaster:
    """
    STL decomposition + ETS on residuals baseline.

    This directly challenges the SignalDecomposer + LSTM architecture:
    both decompose the series the same way, but this uses ETS on residuals
    instead of LSTM. If LSTM beats STL+ETS, the neural network component
    is proven valuable beyond the decomposition alone.

    Steps
    -----
    1. STL decomposes series into trend + seasonal + residual
    2. ETS forecasts the residual
    3. STL trend extrapolated linearly
    4. STL seasonal projected by periodicity
    5. Components recomposed
    """

    def __init__(self, decomposer: SignalDecomposer) -> None:
        self._dc = decomposer
        self._series = None
        self._is_fitted: bool = False

    def fit(self, series: pd.Series) -> "STLETSForecaster":
        self._series = series.copy()
        self._is_fitted = True
        return self

    def forecast(
        self,
        anchor_date: pd.Timestamp,
        n_days: int = FORECAST_HORIZON,
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first.")

        from statsmodels.tsa.stl._stl import STL
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        series_to_anchor = self._series[self._series.index <= anchor_date]
        future_dates = pd.date_range(
            start=anchor_date + pd.Timedelta(days=1), periods=n_days, freq="D"
        )

        try:
            # STL decomposition
            period = min(365, len(series_to_anchor) // 2)
            stl = STL(series_to_anchor, period=period, robust=True)
            stl_fit = stl.fit()

            trend = stl_fit.trend
            seasonal = stl_fit.seasonal
            residual = stl_fit.resid

            # Extrapolate trend linearly from last 90 days
            tail = trend.tail(90)
            x = np.arange(len(tail))
            coeffs = np.polyfit(x, tail.values, 1)
            future_trend = np.array(
                [coeffs[0] * (len(tail) + i) + coeffs[1] for i in range(n_days)]
            )

            # Project seasonal by period
            future_seasonal = np.array(
                [float(seasonal.iloc[-(period - (i % period))]) for i in range(n_days)]
            )

            # ETS on residuals
            try:
                ets_model = ExponentialSmoothing(
                    residual,
                    trend=None,
                    seasonal=None,
                    initialization_method="estimated",
                )
                ets_fit = ets_model.fit(optimized=True)
                future_resid = ets_fit.forecast(n_days).values
            except Exception:
                future_resid = np.zeros(n_days)

            point_forecasts = np.clip(future_trend + future_seasonal + future_resid, 0, None)

        except Exception:
            # Fallback: use SignalDecomposer trend+seasonal + last residual
            trend_vals = self._dc.trend_at(future_dates)
            seasonal_vals = self._dc.seasonal_at(future_dates)
            last_resid = float(
                self._dc.residuals[self._dc.residuals.index <= anchor_date].iloc[-1]
            )
            point_forecasts = np.clip(trend_vals + seasonal_vals + last_resid, 0, None)

        rows = []
        for i, dt in enumerate(future_dates):
            val = float(point_forecasts[i])
            row = {lbl: val for lbl in QUANTILE_LABELS}
            row["date"] = dt
            rows.append(row)

        fan = pd.DataFrame(rows).set_index("date")
        fan.attrs.update({"method": "stl_ets"})
        return fan


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------


def diebold_mariano_test(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: Models A and B have equal forecast accuracy.
    H1: They differ.

    Parameters
    ----------
    losses_a : np.ndarray of per-step loss values for model A
    losses_b : np.ndarray of per-step loss values for model B
    h        : forecast horizon (1 for one-step-ahead)

    Returns
    -------
    (dm_statistic, p_value)
    p_value < 0.05 → reject H0 → difference is statistically significant
    Negative DM statistic → model A is better (lower loss)
    Positive DM statistic → model B is better
    """
    from scipy import stats

    d = losses_a - losses_b
    n = len(d)
    mean_d = np.mean(d)

    # Newey-West HAC variance estimator (accounts for autocorrelation)
    gamma_0 = np.var(d, ddof=1)
    nw_var = gamma_0
    for lag in range(1, h):
        gamma_lag = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        nw_var += 2 * (1 - lag / h) * gamma_lag

    if nw_var <= 0:
        return 0.0, 1.0

    dm_stat = mean_d / np.sqrt(nw_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def run_diebold_mariano(
    eval_results: Dict[str, pd.DataFrame],
    reference_method: str = "lstm",
    quantile: float = 0.50,
) -> pd.DataFrame:
    """
    Run DM test comparing reference_method against all other methods
    for a specific quantile, across all sites.

    Parameters
    ----------
    eval_results     : dict {method: eval_DataFrame} where each DataFrame
                       has columns date, quantile, label, y_true, y_pred,
                       pinball_loss — output of Forecaster.evaluate()
    reference_method : method to test against others (default "lstm")
    quantile         : quantile to test (default 0.50)

    Returns
    -------
    DataFrame with columns:
      site, method_a, method_b, quantile, dm_statistic, p_value, significant
    """
    rows = []
    methods = list(eval_results.keys())
    ref = reference_method

    if ref not in methods:
        raise ValueError(f"reference_method '{ref}' not in eval_results keys: {methods}")

    ref_df = eval_results[ref]
    ref_q = ref_df[np.isclose(ref_df["quantile"], quantile)]

    for method in methods:
        if method == ref:
            continue
        other_df = eval_results[method]
        other_q = other_df[np.isclose(other_df["quantile"], quantile)]

        # Align by date
        common_dates = ref_q["date"].values
        ref_losses = ref_q.set_index("date")["pinball_loss"]
        other_losses = other_q.set_index("date")["pinball_loss"]

        common = ref_losses.index.intersection(other_losses.index)
        if len(common) < 10:
            continue

        dm_stat, p_val = diebold_mariano_test(
            ref_losses.loc[common].values,
            other_losses.loc[common].values,
            h=1,
        )
        rows.append(
            {
                "method_a": ref,
                "method_b": method,
                "quantile": quantile,
                "dm_statistic": round(dm_stat, 4),
                "p_value": round(p_val, 4),
                "significant": "yes" if p_val < 0.05 else "no",
                "winner": ref if dm_stat < 0 else method,
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------
def select_best_model(compare_df: pd.DataFrame, candidates=("lstm", "nbeats")) -> str:
    """
    Pick the winner between two canonical models using MAPE/RMSE and
    stability (lowest variance) across the test-phase evaluation,
    per your resubmission criterion: smallest AND most stable error.
    Expects a DataFrame with columns: method, mape, rmse, error_std.
    """
    sub = compare_df[compare_df["method"].isin(candidates)].copy()
    sub["rank_score"] = sub["rmse"].rank() + sub["mape"].rank() + sub["error_std"].rank()
    winner = sub.sort_values("rank_score").iloc[0]["method"]
    return winner
