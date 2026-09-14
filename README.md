# fhnw_cas_aiops_project1_weather_forcasts

* Projektarbeit: MLOps
* Fachhochschule: FHNW
* Referent: Tobias Mérinat
* Autor: Markus Gerber mit KI-Unterstützung
* Erstellt vom 12.09. - 15.09.2026
* Aufwand: ca. 32h

## Aufgabenstellung

Wir nutzen die Projektarbeit, um die Themen FTI-Architektur (Feature-Training-Inference) und Feature Store vertiefen. Dazu sollen drei Pipelines implementiert werden.

* Feature Store: Als Feature Store verwenden wir den [Hopsworks Feature Store](https://www.hopsworks.ai/), der nach Erstellung eines Accounts gratis genutzt werden kann.
* Model Registry: Als Model Registry kann ebenfalls Hopsworks oder MLflow verwendet werden. Alternativ kann das Modell lediglich lokal gespeichert werden, was im Rahmen der Projektarbeit ebenfalls zulässig ist.
* Umgebung: Die Projektarbeit kann auf GitHub Codespaces (gestartet aus einem eigenen GitHub-Repository) oder lokal auf dem Laptop umgesetzt werden.
* Zielsetzung: Es geht nicht darum, ein hochperformantes Modell zu erstellen. Metriken wie Accuracy, F1-Score usw. des trainierten Modells sind bei dieser Projektarbeit zweitrangig.

---

## Ablauf

1. Wahl einer geeigneten Datenquelle
2. Auswahl eines Targets (Was soll vorhergesagt werden?) und passender Features
3. Erstellung eines GitHub-Repositorys
4. Erstellung eines Hopsworks-Accounts auf [hopsworks.ai](https://www.hopsworks.ai/)
5. Installation der Dependencies (Hinweis: Hopsworks setzt Python < 3.14 voraus)
6. Implementierung der Feature-Pipeline
7. Implementierung der Trainings-Pipeline
8. Implementierung der Inferenz-Pipeline
9. Dokumentation in einer `README.md`-Datei im Repository

---

## Projekt-Details

### Beschreibung
* Dieses Projekt lädt stündliche Wetterdaten über die **Open-Meteo API**, erstellt daraus **Prognose-/Sturmrelevante Features** (Rolling Windows, Druckabfall, Windanomalien etc.) und schreibt die Ergebnisse optional in einen **Hopsworks Feature Store**.
### Zweck
* 🌦️ **Abruf von Wetter-Rohdaten** (past_days + forecast_days) für definierte Standorte  
* 🛠️ **Engineering relevanter Features** für Unwetter-/Sturmindikatoren  
* 🧾 **Aufbau eines finalen Datensatzes** inkl. **Primary Key (event_id)** & **event_time**  
* 🚀 **Upload in eine Hopsworks Feature Group** für spätere ML-Modelle  

### 1. Datenquelle
* Data Source: Open-Meteo Weather API
* Quelle: [https://open-meteo.com/](https://open-meteo.com/)

### 2. Auswahl von Target & Features

#### Target
> „Als Mitarbeiter der Versicherung Die Mobiliar sehe ich in der präzisen Kurzfrist-Prognose für Unwetterwarnungen einen entscheidenden Mehrwert für unsere Risikoprävention.“

#### Features

##### Meteorologisch (Aktuell)
| Feature Name | Datentyp | Beschreibung |
| :--- | :--- | :--- |
| `temperature_2m` | Float | Lufttemperatur in 2 m Höhe (°C) |
| `relative_humidity_2m` | Float | Relative Luftfeuchtigkeit (%) |
| `surface_pressure` | Float | Luftdruck auf Bodenhöhe (hPa) |
| `wind_speed_10m` | Float | Windgeschwindigkeit in 10 m Höhe (km/h) |
| `precipitation` | Float | Niederschlag der letzten Stunde (mm) |

##### Historische Trends (Lags)
| Feature Name | Datentyp | Beschreibung |
| :--- | :--- | :--- |
| `temp_lag_24h` | Float | Temperatur exakt vor 24 Stunden |
| `temp_rolling_avg_7d` | Float | Gleitender Mittelwert der Temperatur (letzte 7 Tage) |

##### Zeit & Saison (Cyclic Features)
| Feature Name | Datentyp | Beschreibung |
| :--- | :--- | :--- |
| `day_of_year_sin` / `day_of_year_cos` | Float | Sinus-/Kosinus-Transformation des Jahrestags (Saisonalität) |
| `hour_of_day` | Integer | Stunde des Tages (0–23) für den Tagesverlauf |

### 3. GitHub-Repository
* Name: `fhnw_cas_aiops_project1_weather_forecasts`

### 4. Hopsworks-Project
* Name: `fhnw_p1_weather_forecasts`
* API-Key sicher laden einrichten mit einer `.env`-Datei via `python-dotenv`

### 5. Dependencies
* Name: `fhnw_p1_weather_forecasts`
* Prüfung der Python-Version (Python < 3.13): `python --version`
  * Installierte Python-Version: 3.14.2
* Installation von Hopsworks: `pip install hopsworks`
* **Erfahrung:** Die Verbindung zu Hopsworks aus Codespaces gestaltete sich schwierig. Ursachen waren:
  * Das fehlende `pyarrow`-Paket
  * Die falsche Python-Version – es funktionierte nur mit Version 3.12

### 6. Feature Pipeline
* Vorgehen: Ich habe die Feature Pipeline Schritt für Schritt aufgebaut und dabei pro Task einen Code-Teil erstellt.
* Folgende Code-Teile habe ich erstellt:

#### 6.1. Hopsworks-Projektverbindungsskript
* Einbau eines sicheren Ladens des API-Keys mit einer `.env`-Datei
* Import der `hopsworks`-Library

#### 6.2. Rohdaten via API abrufen
* Rohdaten via API abrufen, Datenquelle: `url = "https://api.open-meteo.com/v1/forecast"

#### 6.3. Features engineeren
* Features, welche für die Unwetter-Prognosen entscheidend sind:
    * Druckabfall (Vorbote für Stürme)
    * Windböen-Anomalien
    * Rolling Windows für Trends, siehe nachfolgende Tabelle

| Feature | Berechnung | Bedeutung |
|---|---|---|
| Rolling Mean (3h) | Ø Temperatur letzte 3 Stunden | Glatter Trend, weniger Rauschen |
| Rolling Std (6h) | Standardabweichung letzte 6h | Volatilität/Instabilität der Wetterlage |
| Rolling Max/Min (24h) | Max/Min Windgeschwindigkeit letzte 24h | Erkennt Extremwerte im Tagesverlauf |
| Rolling Sum (Niederschlag, 3h) | Summe Regen letzte 3h | Kumulierter Niederschlag → Überflutungsrisiko |
| Delta/Differenz | Aktueller Wert − Rolling Mean | Zeigt Abweichung vom "normalen" Trend |
| Druckänderung (Δp, 3h) | Luftdruck jetzt − Luftdruck vor 3h | Starker Abfall = Hinweis auf Sturm/Unwetter |


#### 6.4. Dataframe und Labels
* Dieses Modul erstellt aus einem Feature-DataFrame (`df`) den **finalen Datensatz** für Training/Inference bzw. für den Upload in ein Feature Store System
* Die Funktion `build_final_dataframe(df)`:
  * selektiert relevante Feature- und Label-Spalten
  * erzeugt einen **Primary Key**: `event_id`
  * setzt die **Event Time**: `event_time`

#### 6.5. 
* Dieser Code erstellt eine **Hopsworks Feature Group** und lädt anschliessend den vorbereiteten `final_df` Datensatz hinein.
* Details der Feature-Group

| Parameter | Wert | Bedeutung |
|---|---|---|
| Name | `weather_features_batch` | Eindeutiger Bezeichner der Feature-Group |
| Version | `1` | Erste Version dieser Feature-Group |
| Primary Key | `event_id` | Eindeutige Identifikation jeder Zeile |
| Event Time | `event_time` | Zeitstempel-Spalte für Time-Travel/Versionierung |
| Time Travel Format | `HUDI` | Speicherformat, das historische Versionen & Time-Travel-Queries ermöglicht |
| Stream | `True` | Ermöglicht Streaming-Inserts (z. B. via Kafka) |
| Description | `"Stündliche Wetterfeatures für die Unwetterprognose"` | Beschreibungstext in der UI |