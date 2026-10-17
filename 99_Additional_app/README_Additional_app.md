# Zusatz: Lokale Web-Demo für das Sturmrisiko (Schweiz)

## 1. Ziel der Erweiterung

Eine kleine, vollständig lokale und kostenlose Weboberfläche (Streamlit, Deutsch). Der Benutzer wählt eine Schweizer Ortschaft, klickt auf **Vorhersage** und sieht das vom bestehenden Modell `model.joblib` vorhergesagte Sturm-/Unwetterrisiko für ca. die nächsten 3 Stunden.

Das Modell `model.joblib` und die ML-/Trainingslogik wurden **nicht verändert**. Die Web-Demo liegt in `99_Additional_app/`. Auf ausdrückliche Freigabe wurde zusätzlich die Standortliste im Projekt vereinheitlicht: neue zentrale Datei `standorte.py` (Projektstamm), angebunden an Feature Pipeline, Inference Pipeline, die Web-Demo und die Dokumentation.

## 2. Architektur

- `storm_logic.py`: Standortliste (aus `../standorte.py`), Wetterabruf, Bereinigung, Feature Engineering, Modell, Risikostufen.
- `app.py`: Streamlit-Oberfläche.
- Externer Dienst (kostenlos, ohne API-Key): Open-Meteo Forecast (dieselbe Datenquelle wie in der Pipeline). Die Geocoding-API wird in der Oberfläche nicht mehr verwendet. Hopsworks wird **nicht** benötigt.

## 3. Ordnerstruktur

(im Projektstamm zusätzlich: `standorte.py` = zentrale Standortliste)
```
99_Additional_app/
├── app.py
├── storm_logic.py
├── requirements.txt
└── README_Additional_app.md
```

## 4. Benötigte Python-Pakete

`streamlit`, `joblib`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `requests` (siehe `requirements.txt`, Versionen wie im Projekt-`requirements.txt`, damit das Modell kompatibel lädt).

## 5. Installation

```bash
cd 99_Additional_app
python -m venv .venv          # .venv ist bereits in .gitignore
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Start der Webanwendung

```bash
cd 99_Additional_app
python -m streamlit run app.py
```

Danach `http://localhost:8501` im Browser öffnen. Die Anwendung läuft nur lokal.

## 7. Laden von model.joblib

`storm_logic.load_model()` lädt per `joblib.load` die Datei
`02_Trainings_Pipeline/severe_weather_model/model.joblib` (nur lesend). Es ist ein `XGBClassifier` mit den Klassen `0` (kein Unwetter) und `1` (Unwetter).

Hinweis: Im Repo existiert auch `01_Feature_Pipeline/severe_weather_model/model.joblib`; sie ist **nicht** identisch (anderer Hash). Verwendet wird die Datei aus der Trainings-Pipeline, da diese dort erzeugt wird (`joblib.dump` in `trainings_pipeline.ipynb`). Pfad bei Bedarf in `MODEL_PATH` (`storm_logic.py`) ändern.

## 8. Benötigte Modell-Features

Quelle: `feature_columns` in `02_Trainings_Pipeline/trainings_pipeline.ipynb`; geprüft gegen `model.feature_names_in_` / Booster (identisch, 22 Features, in dieser Reihenfolge). Latitude/Longitude sind **keine** Modell-Features, sie dienen nur dem Wetterabruf.

| Gruppe | Features |
|---|---|
| Rohdaten (Open-Meteo, stündlich) | temperature_2m, relative_humidity_2m, precipitation, pressure_msl, surface_pressure, cloud_cover, wind_speed_10m, wind_gusts_10m, cape |
| Rolling-Summen Niederschlag | precip_rolling_sum_3h / 6h / 12h |
| Rolling-Maximum Böen | wind_gust_max_3h / 6h / 12h |
| Rolling-Mittel Druck | pressure_mean_3h / 6h / 12h |
| Differenzen | pressure_change_3h, pressure_drop_rate (= change/3), temp_change_3h |
| Anomalie | wind_gust_anomaly (Böe minus 24-h-Mittel) |

Vorverarbeitung (aus der Pipeline übernommen): Grenzwerte pro Variable, Interpolation, Median-Auffüllung, Rolling mit `min_periods=1`, Spalten als numerisch in Modell-Reihenfolge.

Zielvariable: `is_severe_weather` (0/1) = Sturmereignis **3 Stunden später**, wobei Ereignis = Böen > 40 km/h, ODER 3-h-Niederschlagssumme > 10 mm, ODER Druckabfallrate < -0.75 hPa/h.

Nicht eindeutig / Hinweise:
- Die Risikostufen sind keine Modelleigenschaft. Aus der Inference-Pipeline stammen die Schwellen 0.3 (Warnung) und 0.6 (hoch). Die Stufe «Sehr hoch» (>= 0.8) ist eine Festlegung dieser Demo. Anpassbar in `RISK_LEVELS`.
- Der zuletzt dokumentierte Trainingslauf registrierte Modellversion 23. Die lokalen Metriken enthalten keine standortbezogene Aufschlüsselung; eine separate Validierung pro Ort ist nicht belegt. Der Anteil positiver Labels im Trainingssatz beträgt ca. 5.05 %.
- Trainiert wurde mit 30 Tagen Historie und Zeitzone Europe/Zurich; die Inference-Pipeline (und diese Demo) nutzt 1 Tag Historie, UTC. Das ist dort so umgesetzt; Rolling-Features sind dadurch nur nahe am Datenanfang leicht anders.

## 9. Datenfluss

```
Ort (Auswahl aus FALLBACK_PLACES / standorte.py)
  ↓  Name, Kanton, Koordinaten (fest hinterlegt)
Koordinaten (lat/lon)
  ↓  Open-Meteo Forecast (1 Tag Vergangenheit + 3 Tage Vorschau)
Wetterdaten
  ↓  Bereinigung + Feature Engineering (22 Features)
Feature-Zeile der aktuellen Stunde
  ↓
model.joblib (predict_proba / predict)
  ↓
Sturmrisiko (Wahrscheinlichkeit + Stufe)
  ↓
Weboberfläche
```

## 10. Erklärung der Ausgabe

- **Wahrscheinlichkeit**: `predict_proba[:, 1]`, die vom Modell geschätzte Wahrscheinlichkeit eines Unwetterereignisses in ca. 3 Stunden.
- **Risikostufe**: 🟢 Niedrig (< 30 %), 🟡 Mittel (30–60 %), 🟠 Hoch (60–80 %), 🔴 Sehr hoch (>= 80 %).
- **Modellklasse**: `predict()` (interner Modell-Schwellwert 0.5).
- Angezeigt werden außerdem Ort, Kanton, Zeitpunkt der Vorhersage, verwendete Wetterstunde, Horizont und die Feature-Werte.
- Kacheln: 🌬 Böen (`wind_gusts_10m`), 🌧 Regen (`precipitation`), 🌡 Temperatur (`temperature_2m`) und ⚡ Gewitter. Gewitter ist aus `cape` abgeleitet (< 500 niedrig, < 1500 moderat, < 3000 hoch, sonst extrem; Stufen angelehnt an `cape_risk_level` der Feature-Pipeline, keine Modelleigenschaft).
- Die Standortauswahl basiert ausschließlich auf den unter `FALLBACK_PLACES` definierten 10 Ortschaften (Name, Kanton, Koordinaten; keine PLZ vorhanden). `FALLBACK_PLACES` verweist auf die zentrale Liste `standorte.py` im Projektstamm (Single Source of Truth, auch von Feature- und Inference-Pipeline genutzt). Eine freie Ortssuche gibt es in der Oberfläche nicht (`search_places()` bleibt in `storm_logic.py` ungenutzt verfügbar). Generalagenturen der Mobiliar sind nicht hinterlegt.

## 11. Bekannte Einschränkungen

- Keine amtliche Warnung. Die aktuellen lokalen Metriken (F1 0.4452, ROC-AUC 0.9172; siehe `02_Trainings_Pipeline/severe_weather_model/metrics.json`) stammen aus einem einzelnen Train/Test-Split mit ca. 5.05 % positiven Labels im Trainingssatz und belegen keine produktive Warnqualität.
- Internetverbindung für die Wetterdaten nötig (die Standortliste selbst ist fest hinterlegt).
- Open-Meteo ist für nicht-kommerzielle Nutzung kostenlos, mit Abruflimits.
- Die Pipelines verwenden die zehn zentral definierten Standorte. Die gespeicherten Modellmetriken enthalten keine standortbezogene Aufschlüsselung; eine separate Validierung pro Ort ist nicht belegt.

## 12. Kostenhinweis

Es werden **keine** Azure- oder sonstigen kostenpflichtigen Ressourcen erstellt oder gestartet, und es werden keine Deployment-Kommandos ausgeführt. Alles läuft lokal und nutzt nur kostenlose APIs ohne Key.

## 13. Optional: Späterer echter Web Endpoint

- Lokal: `storm_logic.py` hinter FastAPI (`POST /predict` mit Ort oder lat/lon). Kostenlos.
- Streamlit Community Cloud / Hugging Face Spaces: kostenlose Stufen, das Modell müsste im Repo liegen.
- **ACHTUNG: Diese Variante kann Azure-Kosten verursachen.** Azure Managed Online Endpoint (Azure ML): laufende Compute-Kosten, auch ohne Anfragen.
- **ACHTUNG: Diese Variante kann Azure-Kosten verursachen.** Azure Container Apps / App Service / VM / AKS: Kosten je nach Tarif und Laufzeit.

Nichts davon wurde ausgeführt.
