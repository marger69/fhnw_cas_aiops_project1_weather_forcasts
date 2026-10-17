"""Zentrale Standortliste (Single Source of Truth) für das gesamte Projekt.

Verwendet von: 01_Feature_Pipeline, 03_Inference_Pipeline und 99_additional (Web-Demo).
Pro Ort sind Name, Kanton (admin1) und Koordinaten hinterlegt; eine PLZ ist nicht vorhanden.
"""

PLACES = [
    {"name": "Bern", "admin1": "Kanton Bern", "latitude": 46.9480, "longitude": 7.4474},
    {"name": "Zürich", "admin1": "Kanton Zürich", "latitude": 47.3769, "longitude": 8.5417},
    {"name": "Genf", "admin1": "Kanton Genf", "latitude": 46.2044, "longitude": 6.1432},
    {"name": "Lugano", "admin1": "Kanton Tessin", "latitude": 46.0037, "longitude": 8.9511},
    {"name": "Basel", "admin1": "Kanton Basel-Stadt", "latitude": 47.5584, "longitude": 7.5733},
    {"name": "Luzern", "admin1": "Kanton Luzern", "latitude": 47.0502, "longitude": 8.3093},
    {"name": "St. Gallen", "admin1": "Kanton St. Gallen", "latitude": 47.4245, "longitude": 9.3767},
    {"name": "Lausanne", "admin1": "Kanton Waadt", "latitude": 46.5197, "longitude": 6.6323},
    {"name": "Chur", "admin1": "Kanton Graubünden", "latitude": 46.8499, "longitude": 9.5329},
    {"name": "Sion", "admin1": "Kanton Wallis", "latitude": 46.2293, "longitude": 7.3589},
]


def as_location_list():
    """Format der Feature Pipeline: [{'name', 'lat', 'lon'}, ...]."""
    return [{"name": p["name"], "lat": p["latitude"], "lon": p["longitude"]} for p in PLACES]


def as_location_dict():
    """Format der Inference Pipeline: {name: (lat, lon)}."""
    return {p["name"]: (p["latitude"], p["longitude"]) for p in PLACES}
