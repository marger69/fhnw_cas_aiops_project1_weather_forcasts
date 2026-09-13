# fhnw_cas_aiops_project1_weather_forcasts

# Projektarbeit: MLOps

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
* Name: fhnw_cas_aiops_project1_weather_forcasts

### 4. Hopsworks-Project
* Name: fhnw_p1_weather_forcasts
* API-Key sicher laden einrichten mit der .enc Datei mit python-dotenv

### 5. Dependencies
* Name: fhnw_p1_weather_forcasts
* Prüfung der Python-Version (Python < 3.14): `python --version` | Installierte Python-Version: 3.14.2
* Installation von Hopsworks: `pip install hopsworks`
  Es wurde eine virtuelle Umgebung erstellt, weil es Abhängigkeiten gab.