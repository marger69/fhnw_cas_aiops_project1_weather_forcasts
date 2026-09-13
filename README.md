# fhnw_cas_aiops_project1_weather_forcasts

# Projektarbeit: MLOps

## Aufgabenstellung

Wir nutzen die Projektarbeit, um die Themen **FTI-Architektur** (Feature-Training-Inference) und **Featurestore** zu vertiefen. Dazu sollen drei Pipelines implementiert werden.

- **Featurestore:** Als Featurestore verwenden wir den [Hopsworks Feature Store](https://www.hopsworks.ai/), welcher nach Erstellung eines Accounts gratis verwendet werden kann.
- **Model-Registry:** Als Model-Registry kann ebenfalls Hopsworks oder MLflow verwendet werden. Alternativ kann das Modell lediglich lokal gespeichert werden, was im Rahmen der Projektarbeit ebenfalls zulässig ist.
- **Umgebung:** Die Projektarbeit kann auf GitHub Codespaces (gestartet aus einem eigenen GitHub-Repository) oder lokal auf dem Laptop umgesetzt werden.
- **Zielsetzung:** Es geht **nicht** darum, ein hochperformantes Modell zu erstellen. Metriken wie Accuracy, F1-Score etc. des trainierten Modells sind bei dieser Projektarbeit zweitrangig.

---

## Ablauf

1. **Wahl einer geeigneten Datenquelle**
2. **Auswahl eines Targets** (Was soll vorhergesagt werden?) und passender **Features**
3. **Erstellen eines GitHub-Repositorys**
4. **Erstellen eines Hopsworks-Accounts** auf [hopsworks.ai](https://www.hopsworks.ai/)
5. **Installation der Dependencies** (Hinweis: Hopsworks setzt `Python < 3.14` voraus)
6. **Implementation der Feature-Pipeline**
7. **Implementation der Trainings-Pipeline**
8. **Implementation der Inferenz-Pipeline**
9. **Dokumentation** in einem `README.md`-File im Repository

---

## Projekt-Details

### 1. Wahl einer geeigneten Datenquelle
* **Data Source:** Open-Meteo Weather API
* **Quelle:** [https://open-meteo.com/](https://open-meteo.com/)

### 2. Auswahl von Target & Features

#### Target / Motivation
> *„Als Mitarbeiter der Versicherung Die Mobiliar sehe ich in der präzisen Kurzfrist-Prognose für Unwetterwarnungen einen entscheidenden Mehrwert für unsere Risikoprävention.“*

#### Features

##### Meteorologisch (Aktuell)
| Feature Name | Datentyp | Beschreibung |
| :--- | :--- | :--- |
| `temperature_2m` | Float | Lufttemperatur in 2m Höhe (°C) |
| `relative_humidity_2m` | Float | Relative Luftfeuchtigkeit (%) |
| `surface_pressure` | Float | Luftdruck auf Bodenhöhe (hPa) |
| `wind_speed_10m` | Float | Windgeschwindigkeit in 10m Höhe (km/h) |
| `precipitation` | Float | Niederschlag der letzten Stunde (mm) |

##### Historische Trends (Lags)
| Feature Name | Datentyp | Beschreibung |
| :--- | :--- | :--- |
| `temp_lag_24h` | Float | Temperatur exakt vor 24 Stunden |
| `temp_rolling_avg_7d` | Float | Gleitender Mittelwert der Temperatur (letzte 7 Tage) |

##### Zeit & Saison (Cyclic Features)
| Feature Name | Datentyp | Beschreibung |
| :--- | :--- | :--- |
| `day_of_year_sin` / `day_of_year_cos` | Float | Sinus/Kosinus-Transformation des Jahrestags (Saisonalität) |
| `hour_of_day` | Integer | Stunde des Tages (0 – 23) für den Tagesverlauf |




