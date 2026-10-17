"""Lokale Streamlit-Demo: Sturmrisiko für ca. 3 Stunden in der Schweiz."""
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

import storm_logic as sl

st.set_page_config(page_title="Sturmrisiko-Vorhersage Schweiz", page_icon="⛈️", layout="wide")


@st.cache_resource
def get_model():
    return sl.load_model()


def thunder_label(cape: float) -> str:
    # Stufen angelehnt an cape_risk_level der Feature-Pipeline (keine Modelleigenschaft).
    if cape < 500:
        return "niedrig"
    if cape < 1500:
        return "moderat"
    if cape < 3000:
        return "hoch"
    return "extrem"


try:
    model = get_model()
    model_ok = True
except Exception as exc:
    model_ok = False
    model_error = exc

head_left, head_right = st.columns([3, 1])
head_left.title("⛈️ Sturmrisiko-Vorhersage Schweiz")
head_left.caption("Machine Learning Demo")
head_right.markdown("🟢 **Modell bereit**" if model_ok else "🔴 **Modell nicht geladen**")

if not model_ok:
    st.error(f"model.joblib konnte nicht geladen werden ({sl.MODEL_PATH}): {model_error}")
    st.stop()

st.header("Sturmrisiko frühzeitig erkennen")
st.write("KI-basierte Einschätzung für Schweizer Standorte  \nVorhersagehorizont: ca. 3 Stunden")

if "searched" not in st.session_state:
    st.session_state.searched = None

left, right = st.columns(2)

with left:
    with st.container(border=True):
        st.subheader("Standort auswählen")
        places = sl.FALLBACK_PLACES
        place = None
        if places:
            labels = [p["name"] for p in places]
            place = places[st.selectbox("Standort", range(len(places)), format_func=lambda i: labels[i])]
            st.markdown(f"📍 **{place['name']}**  \n{place['admin1']}")
        run = st.button("Risiko berechnen", type="primary", disabled=place is None)

if run and place:
    try:
        with st.spinner("Wetterdaten werden abgerufen …"):
            result = sl.predict_for_location(model, place["latitude"], place["longitude"])
    except Exception as exc:
        st.session_state.searched = None
        st.error(f"Vorhersage fehlgeschlagen: {exc}")
    else:
        st.session_state.searched = {
            "place": place,
            "result": result,
            "time": datetime.now(ZoneInfo("Europe/Zurich")),
        }

state = st.session_state.searched
with right:
    with st.container(border=True):
        st.subheader("STURMRISIKO")
        if state is None:
            st.info("Standort wählen und auf „Risiko berechnen“ klicken.")
        else:
            res, pl = state["result"], state["place"]
            st.markdown(f"<div style='text-align:center;font-size:3rem'>{res['icon']}</div>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center'>{res['label'].upper()}</h2>", unsafe_allow_html=True)
            st.markdown(
                f"<h3 style='text-align:center'>{res['probability'] * 100:.0f} %</h3>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(res["probability"], 0.0), 1.0))
            st.caption("nächste ca. 3 Stunden")
            st.write(f"**{pl['name']}**" + (f" / {pl['admin1']}" if pl["admin1"] else ""))
            st.caption(
                f"Vorhersage vom {state['time'].strftime('%d.%m.%Y %H:%M')} (Europe/Zurich) · "
                f"Wetterstunde {res['data_time'].tz_convert('Europe/Zurich').strftime('%H:%M')} · "
                f"Modellklasse: {'Unwetter erwartet' if res['predicted_class'] else 'kein Unwetter erwartet'}"
            )

if state is not None:
    f = state["result"]["features"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌬 Böen", f"{f['wind_gusts_10m']:.0f} km/h")
    c2.metric("🌧 Regen", f"{f['precipitation']:.1f} mm/h")
    c3.metric("🌡 Temperatur", f"{f['temperature_2m']:.1f} °C")
    c4.metric("⚡ Gewitter", thunder_label(f["cape"]), help=f"CAPE {f['cape']:.0f} J/kg")
    with st.expander("Alle verwendeten Modell-Features"):
        st.dataframe(f.rename("Wert"))

with st.expander("▸ Wie funktioniert die Vorhersage?"):
    st.write(
        "Der gewählte Ort wird in Koordinaten umgewandelt. Dafür werden stündliche Wetterdaten "
        "(Open-Meteo, kostenlos) abgerufen und daraus die 22 Features des trainierten XGBoost-Modells "
        "berechnet. Das Modell schätzt die Wahrscheinlichkeit eines Unwetterereignisses "
        "(starke Böen, viel Regen oder schneller Druckabfall) in ca. 3 Stunden. "
        "Risikostufen: 🟢 unter 30 %, 🟡 30–60 %, 🟠 60–80 %, 🔴 ab 80 %."
    )

st.caption("Prototyp / ML-Demo, keine offizielle Unwetterwarnung.")
