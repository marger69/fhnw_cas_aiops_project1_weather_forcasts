# fhnw_cas_aiops_project1_weather_forcasts

* Projektarbeit: MLOps
* Fachhochschule: FHNW
* Referent: Tobias Mérinat
* Autor: Markus Gerber mit KI-Unterstützung
* Erstellt vom 12.09. bis 15.09.2026
* Aufwand: ca. 32h

## Aufgabenstellung
Wir nutzen die Projektarbeit, um die Themen FTI-Architektur (Feature-Training-Inference) und Feature Store zu vertiefen. Dazu sollen drei Pipelines implementiert werden.

* Feature Store: Als Feature Store verwenden wir den [Hopsworks Feature Store](https://www.hopsworks.ai/), der nach Erstellung eines Accounts gratis genutzt werden kann.
* Model Registry: Als Model Registry kann ebenfalls Hopsworks oder MLflow verwendet werden. Alternativ kann das Modell lediglich lokal gespeichert werden, was im Rahmen der Projektarbeit ebenfalls zulässig ist.
* Umgebung: Die Projektarbeit kann auf GitHub Codespaces (gestartet aus einem eigenen GitHub-Repository) oder lokal auf dem Laptop umgesetzt werden.
* Zielsetzung: Es geht nicht darum, ein hochperformantes Modell zu erstellen. Metriken wie Accuracy, F1-Score usw. des trainierten Modells sind bei dieser Projektarbeit zweitrangig.

### Ablauf
1. Wahl einer geeigneten Datenquelle
2. Auswahl eines Targets (Was soll vorhergesagt werden?) und passender Features
3. Erstellung eines GitHub-Repositorys
4. Erstellung eines Hopsworks-Accounts auf [hopsworks.ai](https://www.hopsworks.ai/)
5. Installation der Dependencies (Hinweis: Hopsworks setzt Python < 3.13 voraus)
6. Implementierung der Feature-Pipeline
7. Implementierung der Trainings-Pipeline
8. Implementierung der Inferenz-Pipeline
9. Dokumentation in einer `README.md`-Datei im Repository

---

## Projekt-Details

### Beschreibung
* Dieses Projekt lädt stündliche Wetterdaten über die **Open-Meteo API**, erstellt daraus **prognose- bzw. sturmrelevante Features** (Rolling Windows, Druckabfall, Windanomalien etc.) und schreibt die Ergebnisse optional in einen **Hopsworks Feature Store**.

### Vorgehen
* Das gesamte Vorgehen orientiert sich an der **FTI-Architektur** (Feature – Training – Inference).

#### Pipeline 1: Feature Pipeline
* Datenbeschaffung
  * Geeignete Wetterdaten über die **Open-Meteo-API** beziehen
  * Relevanten Zeitraum und Standorte definieren

* Datenaufbereitung
  * Rohdaten bereinigen (fehlende Werte, Ausreisser, Duplikate)
  * Daten in ein für die Weiterverarbeitung geeignetes Format bringen

* Feature Engineering
  * Aussagekräftige Wetterfeatures erstellen (Temperatur, Luftdruck, Wind, Niederschlag)
  * Historische Werte und **Rolling Windows** berücksichtigen
  * Zeitliche Merkmale (z. B. Tageszeit, Saison) einbeziehen

* Zieldefinition (Target)
  * Target für die Erkennung von Unwetter- bzw. Sturmereignissen definieren
  * Klare Kriterien/Schwellenwerte für Klassifikation festlegen

* Feature Store
  * Aufbereitete Features in einem **Hopsworks Feature Store** speichern
  * Wiederverwendbarkeit für Training und Inferenz sicherstellen

#### Pipeline 2: Training Pipeline
* Modelltraining & Evaluation
  * Machine-Learning-Modell trainieren
  * Modellqualität mit geeigneten Metriken bewerten (z. B. Precision, Recall, F1-Score)

#### Pipeline 3: Inference Pipeline
* Inferenzpipeline
  * Pipeline zur Verarbeitung neuer Wetterdaten aufbauen
  * Unwetterprognosen in Echtzeit oder nahezu Echtzeit erstellen
---

## Projekt-Dokumentation

### 1. Datenquelle
* Data Source: Open-Meteo Weather API
* Quelle: [https://open-meteo.com/](https://open-meteo.com/)

### 2. Auswahl von Target & Features

#### Target
> „Als Mitarbeiter der Versicherung 'die Mobiliar' sehe ich in der präzisen Kurzfrist-Prognose für Unwetterwarnungen einen entscheidenden Mehrwert für unsere Risikoprävention."

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
* Account: `Mein Google-Account`
* Projektname: `fhnw_cas_aiops_project1_weather_forecasts`

### 4. Hopsworks-Projekt
* Account: `Mein Google-Account`
* Name: `fhnw_p1_weather_forecasts`
* **Bemerkung:** Ich wollte eigentlich den Namen `fhnw_cas_aiops_project1_weather_forecasts` überall durchziehen, die Namenslänge war bei Hopsworks jedoch eingeschränkt.

### 5. Dependencies
* Name der virtuellen Umgebung: `fhnw_p1_weather_forecasts`
* Prüfung der Python-Version (Python < 3.13): `python --version`
* **Bemerkung:** Installierte Python-Version: 3.12.3

### 6. Feature Pipeline
* Vorgehen: Die Feature Pipeline wurde Schritt für Schritt aufgebaut, wobei pro Task ein eigener Code-Teil erstellt wurde.

#### Definitionen
*	7 vergangene Tage (past_days=7)
*	1 zusätzlicher Prognosetag (forecast_days=1)
*	Stündliche Wetterdaten
*	Zeitzone: Europe/Berlin
*	Insgesamt ungefähr 8 Tage Daten
* Definierte Standorte
  *	München, Breitengrad: 48.1351, Längengrad: 11.5820
  * Hamburg, Breitengrad: 53.5511, Längengrad: 9.9937
* Ordnername: Feature Pipeline: `Feature Pipeline`
* Name Jupier-Notebook: `feature_pipeline.ipynb` (pro Task ein Code-Teil)

#### 6.1. Hopsworks-Projektverbindungsskript
* Einbau eines sicheren Ladens des API-Keys mit einer `.env`-Datei
* Installation von Hopsworks: `pip install hopsworks`
* **Bemerkung:** Die Verbindung zu Hopsworks aus Codespaces gestaltete sich schwierig. Die Ursache war ein fehlendes Paket: `pyarrow` (dieses fehlte zunächst und verursachte Verbindungsprobleme).

#### 6.2. Rohdaten via API abrufen
* Datenquelle: `url = "https://api.open-meteo.com/v1/forecast"`
* **Bemerkung:** Sicheres Laden des API-Keys über eine `.env`-Datei mittels `python-dotenv` eingerichtet.

#### 6.3. Feature Engineering
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
* Dieses Modul erstellt aus einem Feature-DataFrame (`df`) den **finalen Datensatz** für Training/Inference bzw. für den Upload in ein Feature-Store-System.
* Die Funktion `build_final_dataframe(df)`:
  * selektiert relevante Feature- und Label-Spalten
  * erzeugt einen **Primary Key**: `event_id`
  * setzt die **Event Time**: `event_time`

#### 6.5. Feature Group erstellen und Daten hochladen
* Dieser Code erstellt eine **Hopsworks Feature Group** und lädt den vorbereiteten `final_df`-Datensatz anschliessend in diese hoch.
* Details der Feature-Group:

| Parameter | Wert | Bedeutung |
|---|---|---|
| Name | `weather_features_batch` | Eindeutiger Bezeichner der Feature-Group |
| Version | `1` | Erste Version dieser Feature-Group |
| Primary Key | `event_id` | Eindeutige Identifikation jeder Zeile |
| Event Time | `event_time` | Zeitstempel-Spalte für Time-Travel/Versionierung |
| Time Travel Format | `HUDI` | Speicherformat, das historische Versionen & Time-Travel-Queries ermöglicht |
| Stream | `True` | Ermöglicht Streaming-Inserts (z. B. via Kafka) |
| Description | `"Stündliche Wetterfeatures für die Unwetterprognose"` | Beschreibungstext in der UI |

#### 6.6. Feature-Pipeline
### Feature Pipeline

Die Funktion `run_feature_pipeline` bildet den zentralen Baustein der Feature-Pipeline. Sie ruft für jede definierte Location Wetterdaten ab, führt das Feature Engineering durch und lädt die Daten optional in den Hopsworks Feature Store hoch.

**Ablauf der Funktion:**

1. **Datenabruf je Standort**
   Für jede Location in der Liste `locations` wird über `fetch_weather_data()` ein DataFrame mit historischen und prognostizierten Wetterdaten abgerufen. Die Parameter `past_days` und `forecast_days` steuern den Zeitraum der Vergangenheits- und Vorhersagedaten. Der Location-Name wird als zusätzliche Spalte ergänzt, um die Daten später zuordnen zu können.

2. **Zusammenführen der Rohdaten**
   Die einzelnen DataFrames aller Standorte werden zu einem gemeinsamen Rohdaten-DataFrame (`raw_df`) zusammengeführt.

3. **Feature Engineering**
   Über `engineer_features()` werden aus den Rohdaten die eigentlichen Wetterfeatures berechnet (z. B. Rolling Windows, Delta-Werte, Druckänderungen, zyklische Zeitmerkmale).

4. **Erstellung des finalen DataFrames**
   `build_final_dataframe()` erzeugt den finalen DataFrame inklusive `event_id` und `event_time`, welcher als Grundlage für Training und Inferenz dient.

5. **Optionaler Upload in den Hopsworks Feature Store**
   Ist der Parameter `upload=True` gesetzt, wird eine Feature Group (`weather_features_batch`) im Hopsworks Feature Store erstellt bzw. referenziert und der finale DataFrame dort eingefügt. Ist kein Feature Store (`fs`) übergeben, wird ein Fehler ausgelöst.

**Parameter der Funktion:**

| Parameter        | Beschreibung                                              |
|-------------------|-------------------------------------------------------------|
| `locations`       | Liste der Standorte mit Namen, Breiten- und Längengrad     |
| `fs`              | Hopsworks Feature Store Objekt (optional, nur bei Upload)   |
| `past_days`       | Anzahl der Tage in der Vergangenheit für den Datenabruf      |
| `forecast_days`   | Anzahl der Vorhersagetage für den Datenabruf                |
| `upload`          | Steuert, ob die Daten in den Feature Store hochgeladen werden |

**Rückgabewerte:**

- `final_df`: Der finale DataFrame mit allen berechneten Features
- `weather_fg`: Referenz auf die Hopsworks Feature Group (nur bei Upload, sonst `None`)

**Hinweis zur Nutzung im Notebook:**

Im gezeigten Beispiel wird die Pipeline mit `upload=False` ausgeführt, da die Feature Group bereits in einer vorherigen Zelle befüllt wurde. Dies ermöglicht eine End-to-End-Prüfung der Pipeline ohne erneuten Upload der Daten.



## Infernece Pipline