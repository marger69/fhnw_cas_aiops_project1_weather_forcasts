# Entwickler-Anleitung

## 1. Zweck und Ergebnis

Dieses Dokument beschreibt, wie die Wetterdaten-Pipeline eingerichtet, ausgeführt und nachvollzogen werden kann. Das Projekt implementiert eine FTI-Architektur:

```text
Open-Meteo API
      |
      v
Feature Pipeline
      |
      v
Hopsworks Feature Store
      |
      v
Trainings-Pipeline und Feature View
      |
      v
XGBoost-Modell und Hopsworks Model Registry
      |
      v
Inference-Pipeline und Unwetterwarnung
```

Das Modell soll für einen aktuellen Wetterzeitpunkt die Wahrscheinlichkeit eines Sturm-/Unwetterereignisses etwa drei Stunden später berechnen. Die Anwendung ist eine MLOps-Demonstration und kein produktives Warnsystem.

## 2. Verwendete Daten

### 2.1 Datenquelle

Die Daten werden über die [Open-Meteo Weather API](https://open-meteo.com/) bezogen. Verwendet werden stündliche Wetterdaten für zwei Beispielstandorte:

| Standort | Breitengrad | Längengrad |
|---|---:|---:|
| München | 48.1351 | 11.5820 |
| Hamburg | 53.5511 | 9.9937 |

Die Feature-Pipeline verwendet historische Daten und einen Forecast. Der im aktuellen Trainingslauf verwendete Abruf umfasst 30 vergangene Tage. Die Inference-Pipeline ruft einen Tag historische Daten sowie drei Forecast-Tage ab, damit die Rolling Features für einen aktuellen Zeitpunkt berechnet und die drei Stunden vorausliegende Zielwahrscheinlichkeit ausgegeben werden können.

Die wichtigsten Rohdatenfelder sind:

- `temperature_2m`: Temperatur in 2 m Höhe
- `relative_humidity_2m`: relative Luftfeuchtigkeit
- `precipitation`: Niederschlag
- `pressure_msl`: Luftdruck auf Meereshöhe
- `surface_pressure`: Luftdruck auf Bodenhöhe
- `cloud_cover`: Bewölkung
- `wind_speed_10m`: Windgeschwindigkeit in 10 m Höhe
- `wind_gusts_10m`: Windböen in 10 m Höhe, direkt von Open-Meteo bezogen
- `cape`: konvektiv verfügbare potenzielle Energie, direkt von Open-Meteo bezogen und auf einen gültigen Wertebereich geprüft

### 2.2 Target

Das Target ist `is_severe_weather` mit zwei Klassen:

- `0`: kein schweres Wetter
- `1`: in etwa drei Stunden erwartetes schweres Wetter beziehungsweise Sturmwarnung

Das Target ist kein extern gemessenes oder amtliches Warnlabel. Es wird in der Feature-Pipeline heuristisch aus Wettermerkmalen erzeugt:

- Windböen über 40 km/h
- Niederschlagssumme der letzten drei Stunden über 10 mm
- Druckänderung pro Stunde unter -0.75 hPa

Sobald mindestens eine Bedingung für die Zielstunde erfüllt ist, wird das Ereignislabel auf `1` gesetzt. Dieses Ereignislabel wird anschliessend je Standort um drei Stunden zurückverschoben. Dadurch bedeutet `is_severe_weather = 1`, dass zum Zeitpunkt des Datensatzes etwa drei Stunden später ein Sturm-/Unwetterereignis erwartet wird.

### 2.3 Feature Engineering

Die Pipeline berechnet zwei Arten von Merkmalen:

1. **Aktuelle beziehungsweise meteorologische Merkmale** wie Temperatur, Luftfeuchtigkeit, Niederschlag, Druck und Wind.
2. **Zeit- und verlaufsbezogene Merkmale** wie Rolling-Summen, Rolling-Maxima, gleitende Druckmittelwerte, Druckänderungen, Windanomalien und Temperaturveränderungen.

Konkrete Modell-Features sind:

```text
temperature_2m
relative_humidity_2m
precipitation
pressure_msl
surface_pressure
cloud_cover
wind_speed_10m
wind_gusts_10m
cape
precip_rolling_sum_3h
precip_rolling_sum_6h
precip_rolling_sum_12h
wind_gust_max_3h
wind_gust_max_6h
wind_gust_max_12h
pressure_mean_3h
pressure_mean_6h
pressure_mean_12h
pressure_change_3h
pressure_drop_rate
wind_gust_anomaly
temp_change_3h
```

Die geplanten zyklischen Zeitmerkmale, `temp_lag_24h` und `temp_rolling_avg_7d` sind im aktuellen Modell nicht enthalten. Massgeblich ist die Feature-Liste in der Trainings-Pipeline.

## 3. Feature-Pipeline

Datei: [01_Feature_Pipeline/feature_pipeline.ipynb](../01_Feature_Pipeline/feature_pipeline.ipynb)

Die Pipeline besteht aus folgenden Schritten:

1. Verbindung mit Hopsworks über `HOPSWORKS_API_KEY` und `HOPSWORKS_PROJECT_NAME`.
2. Abruf der stündlichen Wetterdaten über Open-Meteo.
3. Zusammenführen der Daten für München und Hamburg.
4. Sortieren nach Standort und Zeit.
5. Bereinigung der Rohdaten: Zeitduplikate werden entfernt, physikalisch ungültige Werte als fehlend markiert und fehlende Werte innerhalb des Standorts interpoliert oder mit dem Standortmedian ergänzt.
6. Berechnung der Rolling- und Änderungsfeatures.
7. Erzeugung des heuristischen Ereignislabels und Verschiebung um drei Stunden für `is_severe_weather`.
8. Entfernung der letzten drei Zeilen je Standort, für die kein drei Stunden späterer Zielwert vorhanden ist.
9. Erzeugung der eindeutigen ID `event_id` und des Zeitfelds `event_time`.
10. Upload in die Hopsworks Feature Group `weather_features_batch`, Version `1`.

Die Feature Group verwendet `event_id` als Primary Key und `event_time` als Event Time. Dadurch stehen die berechneten Features zentral und versioniert für die Trainings-Pipeline zur Verfügung.

Die abschliessende Pipeline-Zelle verwendet für den Trainingslauf:

```python
run_feature_pipeline(locations, fs, past_days=30, upload=True)
```

Bei einem erneuten Lauf muss berücksichtigt werden, dass Daten in dieselbe Feature Group geschrieben werden. Die Feature-Pipeline muss vor dem Training erfolgreich abgeschlossen sein.

## 4. Trainings-Pipeline

Datei: [02_Trainings_Pipeline/trainings_pipeline.ipynb](../02_Trainings_Pipeline/trainings_pipeline.ipynb)

Die Trainings-Pipeline führt folgende Schritte aus:

1. Anmeldung am Hopsworks-Projekt.
2. Laden der Feature Group `weather_features_batch`, Version `1`.
3. Auswahl der 22 Modell-Features und des Labels `is_severe_weather`.
4. Erzeugung beziehungsweise Laden der Feature View `severe_weather_fv`, Version `1`.
5. Erstellung eines versionierten Trainingsdatensatzes aus der Feature View.
6. Stratifizierter 80/20-Train-Test-Split mit `random_state=42`.
7. Berechnung eines Klassengewichts für die seltenere Unwetterklasse.
8. Training eines `XGBClassifier`.
9. Evaluation mit Classification Report, F1-Score, ROC-AUC und Confusion Matrix.
10. Speicherung des Modells als `model.joblib` und der Kennzahlen als `metrics.json`, einschliesslich des Zielhorizonts `forecast_horizon_hours: 3`.
11. Upload des Modellordners in die Hopsworks Model Registry.

### 4.1 Modell

Verwendet wird ein `XGBClassifier` aus XGBoost. Das Modell ist für tabellarische Daten geeignet und erhält als Eingabe die oben aufgeführten 22 numerischen Wetter- und Verlaufsfeatures. Das Modell gibt mit `predict_proba` eine Wahrscheinlichkeit für die positive Klasse zurück.

Die lokale Modellablage befindet sich unter:

[02_Trainings_Pipeline/severe_weather_model/](../02_Trainings_Pipeline/severe_weather_model/)

Dort liegen:

- `model.joblib`: serialisiertes XGBoost-Modell
- `metrics.json`: gespeicherte Trainingskennzahlen, Datensatzgrössen und Zielhorizont

Das Modell wird zusätzlich unter dem Namen `severe_weather_classifier` in der Hopsworks Model Registry gespeichert.

## 5. Inference-Pipeline

Datei: [03_Inference_Pipeline/inference_pipeline.ipynb](../03_Inference_Pipeline/inference_pipeline.ipynb)

Die Inference-Pipeline arbeitet in diesen Schritten:

1. Anmeldung an Hopsworks und Laden der Feature Group sowie der versionierten Feature View.
2. Abruf aktueller Wetterdaten und eines kurzen Forecasts von Open-Meteo, einschliesslich `wind_gusts_10m`.
3. Anwendung derselben Duplikat-, Wertebereichs- und Missing-Value-Bereinigung wie in der Feature-Pipeline.
4. Laden der neuesten gespeicherten Batch-Features über die Feature View.
5. Berechnung derselben Rolling-, Druck- und Anomaliefeatures wie beim Training für die aktuellen Live-Daten.
6. Auswahl des aktuellen beziehungsweise nächsten verfügbaren Zeitpunkts als Ausgangspunkt für die Drei-Stunden-Prognose.
7. Ermitteln der höchsten Modellversion aus der Model Registry oder Laden einer über `HOPSWORKS_MODEL_VERSION` festgelegten Version.
8. Laden der Datei `model.joblib`.
9. Prüfung, ob der Live-Feature-Vektor alle vom Modell erwarteten Feature-Namen enthält.
10. Berechnung der Sturmwahrscheinlichkeit mit `predict_proba`.
11. Ableitung von Warnstatus und Risikostufe.

Die Warnschwelle ist standardmässig `0.3`. Eine Wahrscheinlichkeit ab dieser Schwelle erzeugt `storm_warning=True`; ab `0.6` wird zusätzlich die Risikostufe `HOCH` ausgegeben.

Die historischen Batch-Features und das Training stammen aus dem Feature Store. Der aktuelle Inferenz-Feature-Vektor wird zur Laufzeit direkt aus den aktuellen Open-Meteo-Daten berechnet. Das Modell liefert damit die Wahrscheinlichkeit eines Ereignisses etwa drei Stunden nach dem gewählten Ausgangszeitpunkt. Dadurch ist die Feature-Berechnung logisch konsistent, der aktuelle Vektor wird jedoch nicht nochmals als Datensatz in die Feature Group geschrieben.

## 6. Voraussetzungen

### 6.1 Python-Version

Verwendet werden sollte Python **3.12.x**. Im ursprünglichen Projektlauf wurde Python `3.12.3` verwendet. Python 3.13 oder neuer wird wegen der Hopsworks-Abhängigkeiten nicht vorausgesetzt.

Prüfen:

```bash
python --version
```

### 6.2 Virtuelle Umgebung

Im Repository-Stammverzeichnis:

```bash
python -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Unter Windows lautet der Aktivierungsbefehl typischerweise:

```powershell
.venv\Scripts\Activate.ps1
```

### 6.3 Python-Pakete

Die direkten Abhängigkeiten sind in [requirements.txt](../requirements.txt) definiert:

- `hopsworks`: Zugriff auf Feature Store und Model Registry
- `openmeteo-requests`: Abruf der Wetterdaten in der Feature-Pipeline
- `requests-cache`: Zwischenspeicherung von API-Anfragen
- `retry-requests`: Wiederholungslogik bei temporären API-Fehlern
- `python-dotenv`: Laden der Umgebungsvariablen aus `.env`
- `pandas`: DataFrames und Datenaufbereitung
- `numpy`: numerische Berechnungen
- `scikit-learn`: Datensplit, Klassengewichte und Evaluation
- `xgboost`: Klassifikationsmodell
- `joblib`: Speichern und Laden des Modells
- `matplotlib` und `seaborn`: Visualisierung der Evaluation

Die direkten Projektabhängigkeiten sind in `requirements.txt` exakt auf die im Projekt verwendeten Versionen gepinnt. Dazu gehört auch `pyarrow`, das für Hopsworks benötigt wird. Die Python-Version muss weiterhin Python 3.12.x sein; Betriebssystem- und Plattformunterschiede können bei binären Paketen trotzdem abweichende transitive Abhängigkeiten verursachen.

## 7. Zugangsdaten und Konfiguration

Für die Verbindung mit Hopsworks wird im Repository-Stamm eine lokale Datei `.env` benötigt. Diese Datei darf nicht committed werden und ist deshalb in `.gitignore` eingetragen.

Beispiel für den Inhalt, ohne echte Zugangsdaten einzutragen:

```dotenv
HOPSWORKS_API_KEY=<persoenlicher_hopsworks_api_key>
HOPSWORKS_PROJECT_NAME=fhnw_p1_weather_forecasts
```

Der Hopsworks-Projekthost ist im Notebook festgelegt:

```text
eu-west.cloud.hopsworks.ai
```

Die benötigten Zugangsdaten müssen von der ausführenden Person selbst im Hopsworks-Account erzeugt werden. API-Keys dürfen nicht in Notebooks, Git-Historie oder Dokumentation eingetragen werden.

## 8. Ausführungsanleitung

Alle Notebooks sollten aus dem Repository-Stamm oder mit einem korrekt gesetzten Arbeitsverzeichnis geöffnet werden. Die Reihenfolge ist zwingend:

### Schritt 1: Umgebung einrichten

```bash
cd /workspaces/fhnw_cas_aiops_project1_weather_forcasts
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Danach `.env` anlegen und die Hopsworks-Variablen setzen.

### Schritt 2: Feature-Pipeline ausführen

Notebook öffnen:

[01_Feature_Pipeline/feature_pipeline.ipynb](../01_Feature_Pipeline/feature_pipeline.ipynb)

Alle Zellen der Reihe nach ausführen. Die letzte Pipeline-Zelle muss mit `upload=True` laufen. Erwartetes Ergebnis:

- Wetterdaten wurden geladen.
- Features und Labels wurden berechnet.
- Feature Group `weather_features_batch`, Version `1`, wurde befüllt.

### Schritt 3: Trainings-Pipeline ausführen

Notebook öffnen:

[02_Trainings_Pipeline/trainings_pipeline.ipynb](../02_Trainings_Pipeline/trainings_pipeline.ipynb)

Alle Zellen der Reihe nach ausführen. Erwartetes Ergebnis:

- Feature View `severe_weather_fv`, Version `1`, wurde erstellt oder geladen.
- Das Modell wurde trainiert und evaluiert.
- `severe_weather_model/model.joblib` und `metrics.json` wurden erzeugt.
- Das Modell wurde unter `severe_weather_classifier` in die Model Registry hochgeladen.

Falls das Trainingsset nur eine Labelklasse enthält, muss die Feature-Pipeline mit mehr historischen Daten erneut ausgeführt werden.

### Schritt 4: Inference-Pipeline ausführen

Notebook öffnen:

[03_Inference_Pipeline/inference_pipeline.ipynb](../03_Inference_Pipeline/inference_pipeline.ipynb)

Alle Zellen der Reihe nach ausführen. Die Zelle zum Laden des Modells ermittelt standardmässig die höchste vorhandene Registry-Version. Für einen reproduzierbaren Lauf mit einer bestimmten Modellversion kann vor dem Start optional `HOPSWORKS_MODEL_VERSION` in `.env` gesetzt werden, zum Beispiel:

```dotenv
HOPSWORKS_MODEL_VERSION=16
```

Nach einem neuen Training ist dadurch keine Notebook-Anpassung erforderlich; ohne gesetzte Variable wird die höchste vorhandene Version verwendet.

Erwartetes Ergebnis:

```text
Real-Time Ergebnis: {
    'storm_probability': ...,
    'storm_warning': ...,
      'forecast_horizon_hours': 3,
      'forecast_description': 'Sturmrisiko in etwa 3 Stunden',
    'risk_level': ...
}
```

## 9. Nachvollziehbarkeit und Prüfpunkte

Bei einer erfolgreichen Ausführung sollten folgende Objekte vorhanden sein:

| Objekt | Ort beziehungsweise Name |
|---|---|
| Feature-Pipeline | `01_Feature_Pipeline/feature_pipeline.ipynb` |
| Feature Group | Hopsworks: `weather_features_batch`, Version 1 |
| Feature View | Hopsworks: `severe_weather_fv`, Version 1 |
| Trainings-Pipeline | `02_Trainings_Pipeline/trainings_pipeline.ipynb` |
| Modell lokal | `02_Trainings_Pipeline/severe_weather_model/model.joblib` |
| Kennzahlen lokal | `02_Trainings_Pipeline/severe_weather_model/metrics.json` |
| Modell Registry | Hopsworks: `severe_weather_classifier` |
| Inference-Pipeline | `03_Inference_Pipeline/inference_pipeline.ipynb` |

Zusätzlich sollte geprüft werden:

- Die drei Notebooks wurden in der angegebenen Reihenfolge ausgeführt.
- In der Feature Group sind sowohl Labelklasse `0` als auch Labelklasse `1` vorhanden.
- Die Feature-Namen in Training und Inference stimmen überein.
- Die für die Inference verwendete Registry-Modellversion existiert.
- Die `.env`-Datei ist vorhanden, aber nicht Bestandteil des Git-Repositories.

## 10. Bekannte Einschränkungen

- Das Target basiert auf heuristischen Schwellenwerten und nicht auf offiziellen Unwetterwarnungen.
- Die Daten werden bei jedem API-Abruf aktualisiert; dadurch können sich Ergebnisse und Labelverteilung verändern.
- Die aktuelle Inference berechnet den Live-Feature-Vektor direkt aus der API und liest nicht den aktuellen Featurevektor aus dem Feature Store.
- Der Train-Test-Split ist zufallsbasiert. Für eine echte Zeitreihenprognose wäre ein zeitlicher Holdout geeigneter.
- Die direkten Paketversionen sind in `requirements.txt` gepinnt. Für vollständig identische Umgebungen können bei plattformabhängigen Paketen dennoch Unterschiede der transitive Abhängigkeiten auftreten.
- Die Inference ermittelt standardmässig die höchste vorhandene Modellversion. Mit `HOPSWORKS_MODEL_VERSION` kann eine konkrete Version für reproduzierbare historische Läufe festgelegt werden.
- Für Hopsworks ist ein aktiver Account, ein Projekt und ein gültiger API-Key erforderlich.
- Die API sowie Hopsworks müssen während der Ausführung erreichbar sein.

Diese Einschränkungen sind bei der Interpretation der Resultate zu berücksichtigen. Das Projekt demonstriert primär die FTI- und MLOps-Struktur, nicht ein produktionsreifes meteorologisches Warnsystem.
