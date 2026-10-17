"""Logik der Web-Demo: Ort -> Koordinaten -> Wetterdaten -> Features -> model.joblib.

Die Feature-Namen, die Reihenfolge und die Feature-Berechnung wurden aus
01_Feature_Pipeline, 02_Trainings_Pipeline und 03_Inference_Pipeline übernommen.
Der Code der Pipelines wird nicht importiert; nur die zentrale Standortliste (standorte.py) wird gemeinsam genutzt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from standorte import PLACES  # noqa: E402

MODEL_PATH = ROOT / "02_Trainings_Pipeline" / "severe_weather_model" / "model.joblib"

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HORIZON_HOURS = 3
HTTP_TIMEOUT = 20

# Reihenfolge exakt wie `feature_columns` in der Trainings-Pipeline.
FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "pressure_msl",
    "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_gusts_10m", "cape",
    "precip_rolling_sum_3h", "precip_rolling_sum_6h", "precip_rolling_sum_12h",
    "wind_gust_max_3h", "wind_gust_max_6h", "wind_gust_max_12h",
    "pressure_mean_3h", "pressure_mean_6h", "pressure_mean_12h",
    "pressure_change_3h", "pressure_drop_rate", "wind_gust_anomaly", "temp_change_3h",
]
RAW_COLUMNS = FEATURE_COLUMNS[:9]

# Gleiche Plausibilitätsgrenzen wie clean_weather_data() der Pipelines.
VALID_RANGES = {
    "temperature_2m": (-90, 60),
    "relative_humidity_2m": (0, 100),
    "precipitation": (0, 1000),
    "pressure_msl": (850, 1100),
    "surface_pressure": (850, 1100),
    "cloud_cover": (0, 100),
    "wind_speed_10m": (0, 250),
    "wind_gusts_10m": (0, 350),
    "cape": (0, 10000),
}

# Schwellen aus der Inference-Pipeline (threshold=0.3 für Warnung, 0.6 für hoch).
# "Sehr hoch" (>= 0.8) ist eine reine Darstellungs-Konvention dieser Demo.
RISK_LEVELS = [
    (0.8, "🔴", "Sehr hoch"),
    (0.6, "🟠", "Hoch"),
    (0.3, "🟡", "Mittel"),
    (0.0, "🟢", "Niedrig"),
]

# Zentrale Standortliste im Projektstamm (standorte.py), gemeinsam mit den Pipelines genutzt.
FALLBACK_PLACES = PLACES


def load_model(path: Path = MODEL_PATH):
    """Lädt model.joblib (nur lesend)."""
    return joblib.load(path)


def expected_features(model) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = model.get_booster().feature_names
    return list(names)


def search_places(query: str, count: int = 10) -> list[dict]:
    """Sucht Schweizer Orte über die kostenlose Open-Meteo-Geocoding-API."""
    response = requests.get(
        GEOCODING_URL,
        params={"name": query, "count": count, "language": "de", "countryCode": "CH", "format": "json"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    return [
        {
            "name": r["name"],
            "admin1": r.get("admin1", ""),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
        }
        for r in response.json().get("results", [])
        if r.get("country_code") == "CH"
    ]


def fetch_hourly(latitude: float, longitude: float) -> pd.DataFrame:
    """Holt stündliche Rohdaten (1 Tag Vergangenheit + 3 Tage Vorschau, UTC)."""
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(RAW_COLUMNS),
            "past_days": 1,
            "forecast_days": 3,
            "timezone": "UTC",
        },
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Bereinigung wie in der Pipeline: Grenzen, Interpolation, Median-Auffüllung."""
    df = df.sort_values("time").drop_duplicates("time", keep="last").copy()
    for column, (low, high) in VALID_RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce")
        values = values.mask((values < low) | (values > high))
        values = values.interpolate(limit_direction="both")
        df[column] = values.fillna(values.median())
    return df.dropna(subset=RAW_COLUMNS).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-Features identisch zu build_live_feature_vectors() der Inference-Pipeline."""
    df = df.sort_values("time").copy()
    for window in (3, 6, 12):
        df[f"precip_rolling_sum_{window}h"] = df["precipitation"].rolling(window, min_periods=1).sum()
        df[f"wind_gust_max_{window}h"] = df["wind_gusts_10m"].rolling(window, min_periods=1).max()
        df[f"pressure_mean_{window}h"] = df["pressure_msl"].rolling(window, min_periods=1).mean()
    df["pressure_change_3h"] = df["pressure_msl"].diff(3)
    df["pressure_drop_rate"] = df["pressure_change_3h"] / 3
    df["wind_gust_anomaly"] = df["wind_gusts_10m"] - df["wind_gusts_10m"].rolling(24, min_periods=1).mean()
    df["temp_change_3h"] = df["temperature_2m"].diff(3)
    return df


def select_current_row(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.Series:
    """Erste Stunde >= jetzt (wie die Inference-Pipeline); sonst die erste Zeile."""
    now = now or pd.Timestamp.now(tz="UTC")
    candidates = df[df["time"] >= now.floor("h")]
    return (candidates if not candidates.empty else df).iloc[0]


def risk_level(probability: float) -> tuple[str, str]:
    for threshold, icon, label in RISK_LEVELS:
        if probability >= threshold:
            return icon, label
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2]


def predict(model, row: pd.Series) -> dict:
    """Erzeugt X in der vom Modell erwarteten Spaltenreihenfolge und sagt voraus."""
    features = expected_features(model)
    missing = [f for f in features if f not in row.index]
    if missing:
        raise KeyError(f"Fehlende Modell-Features: {missing}")
    X = pd.DataFrame([{f: row[f] for f in features}]).apply(pd.to_numeric, errors="coerce")
    probability = float(model.predict_proba(X)[0, 1])
    icon, label = risk_level(probability)
    return {
        "probability": probability,
        "predicted_class": int(model.predict(X)[0]),
        "icon": icon,
        "label": label,
        "features": X.iloc[0],
    }


def predict_for_location(model, latitude: float, longitude: float) -> dict:
    row = select_current_row(engineer_features(clean(fetch_hourly(latitude, longitude))))
    result = predict(model, row)
    result["data_time"] = row["time"]
    return result
