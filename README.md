# fhnw_cas_aiops_project1_weather_forcasts

* Fachhochschule: FHNW
* CAS: AI Operations
* Projektarbeit: MLOps
* Referent: Tobias Mérinat
* Autor: Markus Gerber
* Erstellt vom 12.09. bis 16.09.2026
* Aufwand: ca. 32h

## Wichtige Links
* Entwickler-Anleitung: [04_Supplementary_information/Entwickler_Anleitung.md](04_Supplementary_information/Entwickler_Anleitung.md)
* Erklärung anhand der FTI-Architektur: [04_Supplementary_information/Erklaerung_anhand_der_FTI_Architektur.md](04_Supplementary_information/Erklaerung_anhand_der_FTI_Architektur.md)
* Feature Pipeline: [01_Feature_Pipeline/feature_pipeline.ipynb](01_Feature_Pipeline/feature_pipeline.ipynb)
* Trainings Pipeline: [02_Trainings_Pipeline/trainings_pipeline.ipynb](02_Trainings_Pipeline/trainings_pipeline.ipynb)
* Inference Pipeline: [03_Inference_Pipeline/inference_pipeline.ipynb](03_Inference_Pipeline/inference_pipeline.ipynb)
* Modell: [02_Trainings_Pipeline/severe_weather_model/model.joblib](02_Trainings_Pipeline/severe_weather_model/model.joblib)
* Metriken: [02_Trainings_Pipeline/severe_weather_model/metrics.json](02_Trainings_Pipeline/severe_weather_model/metrics.json)
* Requirements: [requirements.txt](requirements.txt)
* Data Source Open-Meteo Weather API: [https://open-meteo.com/](https://open-meteo.com/)
* Hopsworks-Projekt: [https://eu-west.cloud.hopsworks.ai/p/44167/view](https://eu-west.cloud.hopsworks.ai/p/44167/view)
* Github-Projekt: [https://github.com/marger69/fhnw_cas_aiops_project1_weather_forcasts](https://github.com/marger69/fhnw_cas_aiops_project1_weather_forcasts)

## 1. Aufgabenstellung
Wir nutzen die Projektarbeit, um die Themen FTI-Architektur (Feature-Training-Inference) und Feature Store zu vertiefen. Dazu sollen drei Pipelines implementiert werden.

* Feature Store: Als Feature Store verwenden wir den [Hopsworks Feature Store](https://www.hopsworks.ai/), der nach Erstellung eines Accounts gratis genutzt werden kann.
* Model Registry: Als Model Registry kann ebenfalls Hopsworks oder MLflow verwendet werden. Alternativ kann das Modell lediglich lokal gespeichert werden, was im Rahmen der Projektarbeit ebenfalls zulässig ist.
* Umgebung: Die Projektarbeit kann auf GitHub Codespaces (gestartet aus einem eigenen GitHub-Repository) oder lokal auf dem Laptop umgesetzt werden.
* Zielsetzung: Es geht nicht darum, ein hochperformantes Modell zu erstellen. Metriken wie Accuracy, F1-Score usw. des trainierten Modells sind bei dieser Projektarbeit zweitrangig.

### 1.1. Ablauf
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

### 1.2. Beschreibung
* Dieses Projekt lädt stündliche Wetterdaten über die **Open-Meteo API**, erstellt daraus **prognose- bzw. sturmrelevante Features** (Rolling Windows, Druckabfall, Windanomalien etc.) und schreibt die Ergebnisse optional in einen **Hopsworks Feature Store**.

### 1.3. Vorgehen
* Das gesamte Vorgehen orientiert sich an der **FTI-Architektur** (Feature – Training – Inference).

#### 1.3.1. Pipeline 1: Feature Pipeline
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

#### 1.3.2. Pipeline 2: Training Pipeline
* Modelltraining & Evaluation
  * Machine-Learning-Modell trainieren
  * Modellqualität mit geeigneten Metriken bewerten (z. B. Precision, Recall, F1-Score)

#### 1.3.3. Pipeline 3: Inference Pipeline
* Inferenzpipeline
  * Pipeline zur Verarbeitung neuer Wetterdaten aufbauen
  * Unwetterprognosen in Echtzeit oder nahezu Echtzeit erstellen
---

## 2. Projekt-Dokumentation

### 2.1. Datenquelle
* Data Source: Open-Meteo Weather API
* Quelle: [https://open-meteo.com/](https://open-meteo.com/)

### 2.2. Auswahl von Target & Features

   * Druckänderung (Δp, 3h) | Luftdruck jetzt − Luftdruck vor 3h | Starker Abfall = Hinweis auf Sturm/Unwetter |
#### 2.2.1. Target
> „Als Mitarbeiter der Mobiliar sehe ich in präzisen Kurzfrist-Prognosen zu Unwetterwarnungen einen entscheidenden Baustein für eine wirksame Risikoprävention."

#### 2.2.2. Features

##### 2.2.2.1. Meteorologisch (Aktuell)
| Feature Name | Datentyp | Beschreibung |
| :--- | :--- | :--- |
| `temperature_2m` | Float | Lufttemperatur in 2 m Höhe (°C) |
| `relative_humidity_2m` | Float | Relative Luftfeuchtigkeit (%) |
| `surface_pressure` | Float | Luftdruck auf Bodenhöhe (hPa) |
| `wind_speed_10m` | Float | Windgeschwindigkeit in 10 m Höhe (km/h) |
| `precipitation` | Float | Niederschlag der letzten Stunde (mm) |

##### 2.2.2.2. Historische Trends (Lags)
| Feature Name | Datentyp | Beschreibung |
| :--- | :--- | :--- |
| `temp_lag_24h` | Float | Temperatur exakt vor 24 Stunden |
| `temp_rolling_avg_7d` | Float | Gleitender Mittelwert der Temperatur (letzte 7 Tage) |

##### 2.2.2.3. Zeit & Saison (Cyclic Features)
| Feature Name | Datentyp | Beschreibung |
| :--- | :--- | :--- |
| `day_of_year_sin` / `day_of_year_cos` | Float | Sinus-/Kosinus-Transformation des Jahrestags (Saisonalität) |
| `hour_of_day` | Integer | Stunde des Tages (0–23) für den Tagesverlauf |

Hinweis: Die oben geplanten zyklischen Zeitfeatures sowie `temp_lag_24h` und
`temp_rolling_avg_7d` sind aktuell nicht Bestandteil des trainierten Modells. Verwendet werden
die in der Feature-Pipeline berechneten Rolling-Summen, Rolling-Maxima, Druckänderungen und
Windanomalien.

### 2.3. GitHub-Repository
* Account: `Mein Account`
* Projektname: `fhnw_cas_aiops_project1_weather_forecasts`

### 2.4. Hopsworks-Projekt
* Account: `Mein Account`
* Name: `fhnw_p1_weather_forecasts`
* Link: https://eu-west.cloud.hopsworks.ai/p/44167/view
* **Bemerkung:** Ich wollte eigentlich den Namen `fhnw_cas_aiops_project1_weather_forecasts` überall durchziehen, die Namenslänge war bei Hopsworks jedoch eingeschränkt.

### 2.5. Dependencies
* Name der virtuellen Umgebung: `fhnw_p1_weather_forecasts`
* Prüfung der Python-Version (Python < 3.13): `python --version`
* **Bemerkung:** Installierte Python-Version: 3.12.3
* Die direkten Projektabhängigkeiten sind in [requirements.txt](requirements.txt) mit exakten Versionen festgelegt. `pyarrow` ist explizit enthalten, da es für die Hopsworks-Anbindung benötigt wird.
* Installation der Abhängigkeiten:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

Die Inference-Pipeline ermittelt standardmässig die höchste vorhandene Version
von `severe_weather_classifier` aus der Hopsworks Model Registry. Für einen
reproduzierbaren Lauf mit einer bestimmten Version kann in `.env` optional
`HOPSWORKS_MODEL_VERSION=16` gesetzt werden. Nach einem neuen Training ist keine
Anpassung des Inference-Notebooks erforderlich.

### 2.6. Feature-Pipeline
* Vorgehen: Die Feature Pipeline wurde Schritt für Schritt aufgebaut, wobei pro Task ein eigener Code-Teil erstellt wurde.

#### 2.6.1. Definitionen
*	7 vergangene Tage (past_days=7)
*	1 zusätzlicher Prognosetag (forecast_days=1)
*	Stündliche Wetterdaten
*	Zeitzone: Europe/Berlin
*	Insgesamt ungefähr 8 Tage Daten
* Tatsächlicher aktueller Trainingslauf: 30 vergangene Tage (`past_days=30`), damit positive und negative Labels entstehen.
* Definierte Standorte
  *	München, Breitengrad: 48.1351, Längengrad: 11.5820
  * Hamburg, Breitengrad: 53.5511, Längengrad: 9.9937
* Ordnername: `Feature Pipeline`
* Name Jupyter-Notebook: `feature_pipeline.ipynb`

#### 2.6.2. Hopsworks-Projektverbindungsskript
* Einbau eines sicheren Ladens des API-Keys mit einer `.env`-Datei
* Installation von Hopsworks: `pip install hopsworks`
* **Bemerkung:** Die Verbindung zu Hopsworks aus Codespaces gestaltete sich schwierig. Die Ursache war ein fehlendes Paket: `pyarrow` (dieses fehlte zunächst und verursachte Verbindungsprobleme).

#### 2.6.3. Rohdaten via API abrufen
* Datenquelle: `url = "https://api.open-meteo.com/v1/forecast"`
* **Bemerkung:** Sicheres Laden des API-Keys über eine `.env`-Datei mittels `python-dotenv` eingerichtet.

#### 2.6.4. Feature Engineering
* Features, welche für die Unwetter-Prognosen entscheidend sind:
    * Druckabfall (Vorbote für Stürme)
    * Windböen-Anomalien
    * Rolling Windows für Trends, siehe nachfolgende Tabelle
* `wind_gusts_10m` wird in der Feature- und Inference-Pipeline direkt über das Open-Meteo-API-Feld bezogen.
* Vor dem Feature Engineering werden Zeitduplikate entfernt, physikalisch ungültige Werte als fehlend markiert und fehlende numerische Werte innerhalb des Standorts interpoliert oder mit dem Standortmedian ergänzt.
* Gültige Extremwerte, beispielsweise starke Windböen, werden nicht als statistische Ausreisser entfernt, da sie für die Unwetterklassifikation relevant sind.

| Feature | Berechnung | Bedeutung |
|---|---|---|
| Rolling Mean (3h) | Ø Temperatur letzte 3 Stunden | Glatter Trend, weniger Rauschen |
| Rolling Std (6h) | Standardabweichung letzte 6h | Volatilität/Instabilität der Wetterlage |
| Rolling Max/Min (24h) | Max/Min Windgeschwindigkeit letzte 24h | Erkennt Extremwerte im Tagesverlauf |
| Rolling Sum (Niederschlag, 3h) | Summe Regen letzte 3h | Kumulierter Niederschlag → Überflutungsrisiko |
| Delta/Differenz | Aktueller Wert − Rolling Mean | Zeigt Abweichung vom "normalen" Trend |
| Druckänderung (Δp, 3h) | Luftdruck jetzt − Luftdruck vor 3h | Starker Abfall = Hinweis auf Sturm/Unwetter |

#### 2.6.5. Dataframe und Labels
* Dieses Modul erstellt aus einem Feature-DataFrame (`df`) den **finalen Datensatz** für Training/Inference bzw. für den Upload in ein Feature-Store-System.
* Die Funktion `build_final_dataframe(df)`:
  * selektiert relevante Feature- und Label-Spalten
  * erzeugt einen **Primary Key**: `event_id`
  * setzt die **Event Time**: `event_time`

#### 2.6.6. Feature Group erstellen und Daten hochladen
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

#### 2.6.7. Feature-Pipeline
Die Funktion `run_feature_pipeline` bildet den zentralen Baustein der Feature-Pipeline. Sie ruft für jede definierte Location Wetterdaten ab, führt das Feature Engineering durch und lädt die Daten optional in den Hopsworks Feature Store hoch.

**Ablauf der Funktion:**
1. **Datenabruf je Standort**
   Für jede Location in der Liste `locations` wird über `fetch_weather_data()` ein DataFrame mit historischen und prognostizierten Wetterdaten abgerufen. Die Parameter `past_days` und `forecast_days` steuern den Zeitraum der Vergangenheits- und Vorhersagedaten. Der Location-Name wird als zusätzliche Spalte ergänzt, um die Daten später zuordnen zu können.

2. **Zusammenführen der Rohdaten**
   Die einzelnen DataFrames aller Standorte werden zu einem gemeinsamen Rohdaten-DataFrame (`raw_df`) zusammengeführt.

3. **Feature Engineering**
  Über `engineer_features()` werden aus den Rohdaten die eigentlichen Wetterfeatures berechnet (z. B. Rolling Windows, Delta-Werte, Druckänderungen und Windanomalien).

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
* `final_df`: Der finale DataFrame mit allen berechneten Features
* `weather_fg`: Referenz auf die Hopsworks Feature Group (nur bei Upload, sonst `None`)

**Hinweis zur Nutzung im Notebook:**
Im gezeigten Beispiel wird die Pipeline mit `upload=False` ausgeführt, da die Feature Group bereits in einer vorherigen Zelle befüllt wurde. Dies ermöglicht eine End-to-End-Prüfung der Pipeline ohne erneuten Upload der Daten.

### 2.7. Trainings-Pipeline
#### 2.7.1. Definitionen
* Ordnername: `Training Pipeline`
* Name Jupyter-Notebook: `training_pipeline.ipynb`

#### 2.7.2. Feature Group laden
* Login bei Hopsworks über `hopsworks.login()`
  * Laden der `.env`-Datei mit den Zugangsdaten
  * Auslesen von API-Key und Projektname aus den Umgebungsvariablen
* Zugriff auf den Feature Store via `project.get_feature_store()`
* Laden der Feature Group `weather_features_batch` (Version 1)
* Ausgabe von Name, Version und Anzahl Features zur Bestätigung
* Bestehende Feature Group `weather_features_batch` 

#### 2.7.3. Feature View erstellen (Query mit Features + Label)
* Definiert eine Liste von 22 Feature-Spalten (u. a. Temperatur, Luftfeuchtigkeit, Druck, Wind, rollende Fenster-Aggregationen) sowie die Ziel-/Label-Spalte `is_severe_weather`
* Prüft anhand der verfügbaren Spalten in `weather_fg`, ob alle benötigten Features und das Label tatsächlich vorhanden sind
* Löst bei fehlenden Spalten einen `ValueError` mit genauer Auflistung aus, um inkonsistente Feature Views zu verhindern
* Erstellt über `weather_fg.select()` eine Query, die nur die relevanten Feature- und Label-Spalten enthält
* Legt mittels `fs.get_or_create_feature_view()` den Feature View `severe_weather_fv` (Version 1) an oder ruft ihn ab, falls er bereits existiert
* Verknüpft die Query mit dem Label `is_severe_weather`, um den Feature View direkt für überwachtes Lernen nutzbar zu machen
* Ermöglicht durch den Feature View eine reproduzierbare und versionierte Grundlage für Trainings- und Testdatensätze
* Gibt nach erfolgreicher Erstellung Name und Version des Feature Views zur Bestätigung aus

#### 2.7.4. Modell trainieren und evaluieren
* Erstellt aus dem Feature View `severe_weather_fv` einen versionierten Trainingsdatensatz im CSV-Format über `create_training_data()`
* Lädt die Trainingsdaten (Features `X` und Label `y`) mittels `get_training_data()` und wandelt die Zielspalte `is_severe_weather` in einen numerischen, ganzzahligen Typ um
* Teilt die Daten stratifiziert in Trainings- und Testset auf (80/20), um die Klassenverteilung in beiden Sets beizubehalten
* Berechnet ausgeglichene Klassengewichte mit `compute_class_weight()`, um dem Ungleichgewicht zwischen seltenen Unwetter-Ereignissen und Normalfällen entgegenzuwirken
* Konfiguriert einen `XGBClassifier` mit angepasstem `scale_pos_weight`, um die Vorhersage der Minderheitsklasse (Unwetter) zu verbessern
* Trainiert das Modell auf den Trainingsdaten und nutzt das Testset als Evaluationsset während des Trainings (`eval_set`)
* Erstellt Vorhersagen auf dem Testset und wertet das Modell mittels `classification_report()` aus (Precision, Recall, F1-Score je Klasse)
* Dient als zentraler Trainingsschritt der Trainings-Pipeline, um ein einsatzfähiges Klassifikationsmodell für Unwetterereignisse zu erzeugen

#### 2.7.5. Modell evaluieren
* Erstellt Vorhersagen (`y_pred`) sowie Wahrscheinlichkeiten (`y_pred_proba`) für die positive Klasse auf Basis des Testsets
* Gibt einen ausführlichen `classification_report` mit Precision, Recall und F1-Score für die Klassen "Kein Unwetter" und "Unwetter" aus
* Berechnet den F1-Score als zusammenfassende Kennzahl zur Bewertung der Modellgüte bei unausgeglichenen Klassen
* Berechnet den ROC-AUC-Wert, sofern im Testset beide Klassen vorhanden sind, um die Trennschärfe des Modells zu bewerten
* Fängt den Sonderfall ab, dass im Testset nur eine Klasse vorkommt, und gibt in diesem Fall eine entsprechende Warnung anstelle des ROC-AUC-Werts aus
* Erstellt eine Confusion Matrix zur Visualisierung von richtig- und falsch-klassifizierten Fällen je Klasse
* Visualisiert die Confusion Matrix als Heatmap mittels `seaborn`, inklusive Achsentiteln für tatsächliche und vorhergesagte Klassen
* Speichert die Confusion-Matrix-Grafik als PNG-Datei (`confusion_matrix.png`) zur Dokumentation der Modellleistung
* Dient als abschliessender Evaluationsschritt, um die Eignung des Modells für die Unwettererkennung nachvollziehbar zu belegen

#### 2.7.6. Feature Importance
* Berechnet die Wichtigkeit jedes Features anhand von `model.feature_importances_`
* Sortiert die Features absteigend nach ihrer Relevanz für das Modell
* Gibt die Top 10 wichtigsten Features in der Konsole aus
* Visualisiert die Top 10 Features als Balkendiagramm mittels `seaborn`
* Speichert die Grafik als `feature_importance.png` zur Dokumentation
* Unterstützt die Interpretierbarkeit des Modells im Rahmen der Trainings-Pipeline

#### 2.7.7. Modell speichern und Metriken für die Model Registry vorbereiten
* Legt ein lokales Verzeichnis `severe_weather_model` zur Ablage von Modell und Metriken an
* Speichert das trainierte Modell mittels `joblib` als `model.joblib`
* Sammelt relevante Kennzahlen wie F1-Score, Anzahl Trainings- und Testdaten sowie den Anteil positiver Klassen
* Prüft, ob der ROC-AUC-Wert definiert ist, und schliesst ihn nur bei Gültigkeit in die Metriken ein
* Gibt eine Warnung aus, falls der ROC-AUC-Wert nicht bestimmbar ist
* Speichert die Metriken strukturiert als `metrics.json`-Datei
* Bereitet damit Modell und Metriken für eine spätere Registrierung in der Model Registry vor

#### 2.7.8. Upload in Hopsworks Model Registry
* Ruft die Model Registry des Hopsworks-Projekts über `project.get_model_registry()` ab
* Definiert Input- und Output-Schema anhand der Trainingsdaten zur Dokumentation und Validierung
* Kombiniert beide Schemas in einem `ModelSchema` für eine vollständige Modellbeschreibung
* Erstellt ein neues Modell-Objekt in der Registry mit Name, Metriken, Schema und Beispiel-Input
* Verknüpft das Modell mit der zugehörigen Feature View, um die Lineage nachvollziehbar zu machen
* Lädt das lokal gespeicherte Modellverzeichnis in die Model Registry hoch
* Gibt nach erfolgreichem Upload Name, Version und Model-ID zur Bestätigung aus

### 2.8. Inference-Pipeline
#### 2.8.1. Definitionen
* 1 vergangener Tag für die Rolling-Window-Berechnung
*	3 Prognosetage (forecast_days=3)
*	Stündliche Live-Wetterdaten
*	Zeitzone: UTC
* Ordnername: `Inference Pipeline`
* Name Jupyter-Notebook: `feature_inference.ipynb`

#### 2.8.2. Hopsworks Feature Group laden
* Login bei Hopsworks über `hopsworks.login()`
  * Laden der `.env`-Datei mit den Zugangsdaten
  * Auslesen von API-Key und Projektname aus den Umgebungsvariablen
* Greift auf den zugehörigen Feature Store des Projekts zu
* Lädt die Feature Group `weather_features_batch` in der Version 1
* Validiert, ob die Feature Group erfolgreich geladen wurde, und bricht andernfalls ab
* Bestätigt den erfolgreichen Ladevorgang mit Name und Version der Feature Group

#### 2.8.3. Live-Daten abrufen (Forecast von Open-Meteo)
* Definiert die Funktion `fetch_live_forecast`, um aktuelle Wetterdaten sowie eine 3-Tage-Vorhersage von der Open-Meteo API abzurufen
* Fragt relevante stündliche Wetterparameter ab, darunter Temperatur, Luftfeuchtigkeit, Niederschlag, Druck, Wind und CAPE
* Berücksichtigt zusätzlich einen Tag in der Vergangenheit (`past_days`), um Rolling-Window-Berechnungen zu ermöglichen
* Wandelt die API-Antwort in einen strukturierten `pandas.DataFrame` um und ergänzt Standortinformationen
* Nutzt `response.raise_for_status()`, um Fehler bei der API-Abfrage frühzeitig zu erkennen
* Gibt die Anzahl geladener Datenpunkte pro Standort zur Kontrolle aus
* Definiert die Standorte München und Hamburg mit ihren jeweiligen Koordinaten als Beispiel für die Live-Abfrage

#### 2.8.4. User-Input für Ad-hoc-Abfrage (Optional)
* Definiert die Funktion `get_user_location_input`, um eine benutzerdefinierte Standortabfrage zu ermöglichen
* Fragt Breiten- und Längengrad interaktiv über die Konsole ab
* Erfasst zusätzlich einen frei wählbaren Ortsnamen zur besseren Zuordnung
* Wandelt die eingegebenen Koordinaten in numerische Werte (`float`) um
* Gibt die eingegebenen Standortdaten als strukturiertes Dictionary zurück
* Ermöglicht dadurch flexible Ad-hoc-Abfragen für beliebige Standorte ausserhalb der vordefinierten Liste

#### 2.8.5. Real-Time Modus (Online Feature Store)
* Ruft die aktuelle Open-Meteo-Abfrage über `fetch_live_forecast()` tatsächlich auf
* Berechnet aus den Live-Daten dieselben Rolling-, Druck- und Anomalie-Features wie die Feature-Pipeline
* Wählt den aktuellen bzw. nächsten verfügbaren Zeitpunkt aus und erstellt daraus einen Feature-Vektor
* Verwendet diesen Live-Feature-Vektor direkt für die Prediction
* Der Featurestore bleibt die Quelle für historische Batch-Features und das Training; das aktuelle Feature wird zur Inferenzzeit aus der API bezogen

#### 2.8.6. Modell aus der Model Registry herunterladen
* Greift über `project.get_model_registry()` auf die Hopsworks Model Registry zu
* Ruft das Modell `severe_weather_classifier` in der angegebenen Version ab (alternativ automatisch die neueste Version)
* Lädt das zugehörige Modellverzeichnis lokal über `model_meta.download()` herunter
* Bestätigt den erfolgreichen Download mit Ausgabe des lokalen Speicherpfads
* Lädt das serialisierte Modell (`model.joblib`) mittels `joblib.load()` in den Arbeitsspeicher
* Gibt Name und Version des geladenen Modells zur Kontrolle aus
* Zeigt die im Model Registry gespeicherten Trainings-Metriken zur Nachvollziehbarkeit an

#### 2.8.7. Real-Time Single Prediction
* Definiert die Funktion `run_realtime_prediction`, um für einen einzelnen Feature-Vektor eine Echtzeit-Vorhersage durchzuführen
* Ermittelt die vom Modell erwarteten Feature-Namen über `feature_names_in_` oder alternativ über den XGBoost-Booster
* Prüft, ob alle vom Modell benötigten Features im übergebenen Feature-Vektor vorhanden sind, und bricht bei fehlenden Werten mit `KeyError` ab
* Baut aus dem Feature-Vektor einen `pandas.DataFrame` auf und wandelt alle Werte numerisch um
* Berechnet die Sturmwahrscheinlichkeit mittels `model.predict_proba()` und leitet daraus eine binäre Sturmwarnung anhand eines Schwellenwerts ab
* Klassifiziert das Risiko zusätzlich in die Stufen „HOCH“, „MITTEL“ oder „NIEDRIG“ zur besseren Interpretierbarkeit
* Gibt das Ergebnis als strukturiertes Dictionary mit Wahrscheinlichkeit, Warnstatus und Risikostufe zurück
* Führt die Funktion beispielhaft mit dem zuvor geladenen Live-Feature-Vektor aus und gibt das Resultat aus
---

## 3. Persönliche Konklusion
* Durch dieses Projekt konnte ich die FTI-Architektur mit den drei Bereichen **Feature, Training und Inference** praktisch umsetzen und besser verstehen.
* Ich habe gelernt, wie Wetterdaten über eine API abgerufen, aufbereitet und für ein Machine-Learning-Modell weiterverwendet werden können.
* Besonders interessant war für mich das Feature Engineering mit Rolling Windows, Druckveränderungen, Windanomalien und Niederschlagssummen.
* Mit Hopsworks konnte ich einen Feature Store sowie eine Model Registry kennenlernen und die Wiederverwendung von Features und Modellen umsetzen.
* Die Anbindung an Hopsworks war teilweise anspruchsvoll, insbesondere wegen der benötigten Python-Version und des fehlenden `pyarrow`-Pakets.
* Die Trennung in eigene Pipelines macht den Ablauf übersichtlicher und ermöglicht es, Feature-Erstellung, Training und Vorhersage unabhängig voneinander weiterzuentwickeln.
* Ich habe erkannt, wie wichtig eine konsistente Feature-Erstellung ist, damit Training und Inference mit derselben Datenstruktur arbeiten.
* Das Projekt hat mir gezeigt, dass neben dem Modell selbst auch Datenqualität, Konfiguration, Versionsverwaltung und reproduzierbare Abläufe entscheidend sind.
* Die Modellqualität stand in dieser Arbeit nicht im Vordergrund. Trotzdem konnte ich den vollständigen Ablauf von der Datenbeschaffung bis zur Unwetterprognose realisieren.
* Als mögliche Weiterentwicklung sehe ich die Verwendung historischer Unwetterdaten, eine genauere Definition des Targets und eine grössere Anzahl an Standorten, um das Modell realistischer zu trainieren.

### 3.1. Limitationen und Ausführungshinweise
* Das Target wird aus heuristischen Wetter-Schwellenwerten erzeugt und ist kein extern validiertes
  Unwetterlabel. Die historischen Daten müssen mindestens eine positive und eine negative
  Labelklasse enthalten.
* Das Training prüft diese Voraussetzung und bricht mit einer verständlichen Fehlermeldung ab,
  wenn nur eine Klasse vorhanden ist. Nach einer Änderung der Daten oder Schwellenwerte müssen
  Feature- und Trainings-Pipeline erneut ausgeführt und das Modell neu registriert werden.
* Das Training verwendet aktuell einen stratifizierten Zufallssplit. Für einen produktiven
  Wetterforecast wäre zusätzlich ein zeitlicher Holdout sinnvoll.
* Ausführungsreihenfolge: zuerst `Feature Pipeline/feature_pipeline.ipynb` mit `upload=True`,
  danach `Trainings Pipeline/trainings_pipeline.ipynb` und zuletzt
  `Inference Pipeline/inference_pipeline.ipynb`.