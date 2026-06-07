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
  Input window : 60 days of noise residuals
  Forecast horizon : 15 days ahead
  Models    : 12 total (6 sites × 2 variables)

Usage
-----
  # Phase 1 — empirical quantiles only (no neural net required):
  fcast = Forecaster(df, variable="solar")
  fcast.fit_empirical()
  fan = fcast.forecast_empirical(site="Natal", n_days=15)

  # Phase 2 — LSTM on top:
  fcast.fit_lstm(site="Natal", epochs=50)
  fan = fcast.forecast_lstm(site="Natal", n_days=15)

  # Convenience — auto-choose best available model:
  fan = fcast.forecast(site="Natal", n_days=15)

Requirements
------------
  Phase 1 : numpy, pandas, scipy               (always available)
  Phase 2 : torch (pip install torch)           (optional)

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
INPUT_WINDOW: int = 60
FORECAST_HORIZON: int = 15
MIN_TRAIN_DAYS: int = 730


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
            row = {lbl: base + q_dict[q] for q, lbl in zip(QUANTILES, QUANTILE_LABELS)}
            row["date"] = future_dates[i]
            rows.append(row)

        df = pd.DataFrame(rows).set_index("date")
        return df.clip(lower=0.0)


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
    def __init__(
        self,
        input_window: int = INPUT_WINDOW,
        horizon: int = FORECAST_HORIZON,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        if not _try_import_torch():
            raise ImportError(
                "PyTorch is required for LSTM training. "
                "Install with: pip install torch"
            )
        import torch
        import torch.nn as nn

        self.input_window = input_window
        self.horizon = horizon
        self.n_quantiles = len(QUANTILES)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Net(nn.Module):
            def __init__(self, hidden: int, layers: int, drop: float, nq: int, h: int):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    dropout=drop if layers > 1 else 0.0,
                )
                self.head = nn.Linear(hidden, nq * h)
                self.nq = nq
                self.h = h

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                raw = self.head(last)
                return raw.view(-1, self.nq, self.h)

        self._net = _Net(hidden_size, num_layers, dropout, self.n_quantiles, horizon)
        self._net.to(self._device)
        self._scaler_mean: float = 0.0
        self._scaler_std: float = 1.0
        self._is_trained: bool = False

    def _pinball_loss(self, preds, targets):
        import torch
        total = torch.zeros(1, device=self._device)
        for qi, q in enumerate(QUANTILES):
            err = targets - preds[:, qi, :]
            loss = torch.where(err >= 0, q * err, (q - 1) * err)
            total = total + loss.mean()
        return total / len(QUANTILES)

    def _make_sequences(
        self, residuals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        T = self.input_window
        H = self.horizon
        n = len(residuals)
        for start in range(n - T - H + 1):
            X.append(residuals[start : start + T])
            Y.append(residuals[start + T : start + T + H])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def fit(
        self,
        residuals: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        val_frac: float = 0.15,
        patience: int = 10,
        verbose: bool = True,
    ) -> List[float]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._scaler_mean = float(np.mean(residuals))
        self._scaler_std = float(np.std(residuals)) or 1.0
        scaled = (residuals - self._scaler_mean) / self._scaler_std

        X, Y = self._make_sequences(scaled)
        n_val = max(1, int(len(X) * val_frac))
        X_tr, Y_tr = X[:-n_val], Y[:-n_val]
        X_val, Y_val = X[-n_val:], Y[-n_val:]

        X_tr_t = torch.tensor(X_tr[:, :, None]).to(self._device)
        Y_tr_t = torch.tensor(Y_tr).to(self._device)
        X_val_t = torch.tensor(X_val[:, :, None]).to(self._device)
        Y_val_t = torch.tensor(Y_val).to(self._device)

        ds = TensorDataset(X_tr_t, Y_tr_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(self._net.parameters(), lr=lr)

        val_losses: List[float] = []
        best_val = math.inf
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self._net.train()
            for xb, yb in loader:
                optim.zero_grad()
                preds = self._net(xb)
                loss = self._pinball_loss(preds, yb.unsqueeze(1).expand_as(preds))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optim.step()

            self._net.eval()
            with torch.no_grad():
                val_preds = self._net(X_val_t)
                val_loss = self._pinball_loss(
                    val_preds, Y_val_t.unsqueeze(1).expand_as(val_preds)
                )
            vl = float(val_loss.item())
            val_losses.append(vl)

            if verbose and epoch % 10 == 0:
                logger.info("Epoch %3d/%d  val_pinball=%.5f", epoch, epochs, vl)

            if vl < best_val:
                best_val = vl
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stop at epoch %d (patience=%d)", epoch, patience)
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._is_trained = True
        return val_losses

    def predict(self, last_residuals: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Call .fit() before .predict().")
        import torch

        scaled = (last_residuals - self._scaler_mean) / self._scaler_std
        x = torch.tensor(scaled[None, :, None], dtype=torch.float32).to(self._device)
        self._net.eval()
        with torch.no_grad():
            out = self._net(x)
        out_np = out.squeeze(0).cpu().numpy()
        return out_np * self._scaler_std + self._scaler_mean

    def save(self, path: Path) -> None:
        import torch
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "scaler_mean": self._scaler_mean,
                "scaler_std": self._scaler_std,
            },
            path,
        )

    def load(self, path: Path) -> None:
        import torch
        ckpt = torch.load(path, map_location=self._device)
        self._net.load_state_dict(ckpt["state_dict"])
        self._scaler_mean = ckpt["scaler_mean"]
        self._scaler_std = ckpt["scaler_std"]
        self._is_trained = True


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
                f"No data found for site='{site}', variable='{self.variable}'."
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
            eq = EmpiricalQuantileForecaster(dc).fit()
            self._emp_forecasters[site] = eq
            logger.info(
                "Empirical forecaster fitted for %s (%d days).", site, len(series)
            )

        self._empirical_fitted = True
        return self

    def fit_lstm(
        self,
        site: str,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        val_frac: float = 0.15,
        patience: int = 10,
        verbose: bool = True,
        force_retrain: bool = False,
    ) -> List[float]:
        if not self._empirical_fitted:
            raise RuntimeError("Call fit_empirical() before fit_lstm().")
        if site not in self._decomposers:
            raise ValueError(f"Site '{site}' not fitted empirically yet.")

        model_path = (
            self._model_dir
            / f"lstm_{self.variable}_{site.replace(' ', '_')}.pt"
        )
        model = _QuantileLSTMModel()

        if model_path.exists() and not force_retrain:
            logger.info("Loading existing LSTM from %s", model_path)
            model.load(model_path)
            self._lstm_models[site] = model
            self._lstm_fitted_sites.append(site)
            return []

        residuals = self._decomposers[site].residuals.values
        if len(residuals) < MIN_TRAIN_DAYS:
            raise ValueError(
                f"Site '{site}' has only {len(residuals)} days; "
                f"need >= {MIN_TRAIN_DAYS} to train LSTM."
            )

        val_losses = model.fit(
            residuals,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            val_frac=val_frac,
            patience=patience,
            verbose=verbose,
        )
        model.save(model_path)
        self._lstm_models[site] = model
        if site not in self._lstm_fitted_sites:
            self._lstm_fitted_sites.append(site)
        return val_losses

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
        fan.attrs["site"] = site
        fan.attrs["variable"] = self.variable
        fan.attrs["method"] = "empirical"
        return fan

    def forecast_lstm(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if site not in self._lstm_models:
            raise RuntimeError(
                f"LSTM not trained for site '{site}'. "
                f"Call fit_lstm('{site}') first."
            )

        dc = self._decomposers[site]
        model = self._lstm_models[site]
        series = self._get_site_series(site)

        if anchor_date is None:
            anchor_date = series.index[-1]

        resid = dc.residuals
        context = resid[resid.index <= anchor_date].values[-INPUT_WINDOW:]
        if len(context) < INPUT_WINDOW:
            pad = np.zeros(INPUT_WINDOW - len(context))
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
        fan.attrs["site"] = site
        fan.attrs["variable"] = self.variable
        fan.attrs["method"] = "lstm"
        return fan

    def forecast(
        self,
        site: str,
        n_days: int = FORECAST_HORIZON,
        anchor_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if site in self._lstm_models:
            return self.forecast_lstm(site, n_days=n_days, anchor_date=anchor_date)
        return self.forecast_empirical(site, n_days=n_days, anchor_date=anchor_date)

    def evaluate(
        self,
        site: str,
        method: str = "empirical",
        test_days: int = 90,
    ) -> pd.DataFrame:
        series = self._get_site_series(site)
        n = len(series)
        if n < INPUT_WINDOW + test_days + 1:
            raise ValueError("Not enough data for evaluation.")

        cut_idx = n - test_days
        records = []

        for step in range(test_days):
            anchor = series.index[cut_idx + step - 1]
            true_val = float(series.iloc[cut_idx + step])
            if method == "lstm" and site in self._lstm_models:
                fan = self.forecast_lstm(site, n_days=1, anchor_date=anchor)
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

    Example
    -------
    >>> from src.nasa_loader import load_combined_csv, quality_filter
    >>> from src.nasa_loader import gap_fill, add_hub_height_wind
    >>> from src.forecaster import build_forecasters
    >>> params = load_params()
    >>> df = load_combined_csv(params)
    >>> df = quality_filter(df); df = gap_fill(df)
    >>> df = add_hub_height_wind(df, params)
    >>> forecasters = build_forecasters(df)
    >>> fan = forecasters["solar"].forecast(site="Natal")
    """
    forecasters: Dict[str, Forecaster] = {}
    for var in VARIABLES:
        f = Forecaster(df, variable=var, model_dir=model_dir)
        if fit_empirical:
            f.fit_empirical()
        forecasters[var] = f
    return forecasters


# ---------------------------------------------------------------------------
# ForecastBundle — joint solar + wind fan with correlated quantiles
# ---------------------------------------------------------------------------

class ForecastBundle:
    """
    Joint operational forecast for one site.
    Attaches seasonal solar-wind correlation coefficient (rho) to each
    forecast row so diesel_model.py can propagate joint worst-case quantiles.
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
        bundle.attrs["site"] = site
        bundle.attrs["n_days"] = n_days
        return bundle
