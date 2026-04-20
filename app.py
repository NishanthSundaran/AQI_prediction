"""AQI Prediction & Forecasting for Chennai.

Streamlit front-end combining:
- SARIMA-based monthly forecasts of seven pollutants
  (AQI, CO, NO2, O3, PM10, PM2.5, SO2)
- CPCB India sub-index AQI calculator that converts
  pollutant concentrations into an overall Air Quality Index.
"""

import pickle
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------- config

st.set_page_config(
    page_title="AQI Prediction — Chennai",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path(__file__).parent / "models"

POLLUTANTS = {
    "AQI":   {"file": "aqi.pkl",  "unit": "index"},
    "CO":    {"file": "co.pkl",   "unit": "mg/m³"},
    "NO2":   {"file": "no2.pkl",  "unit": "µg/m³"},
    "O3":    {"file": "o3.pkl",   "unit": "µg/m³"},
    "PM10":  {"file": "pm10.pkl", "unit": "µg/m³"},
    "PM2.5": {"file": "pm25.pkl", "unit": "µg/m³"},
    "SO2":   {"file": "so2.pkl",  "unit": "µg/m³"},
}

# CPCB India breakpoint table.
# Each row: (C_LO, C_HI, I_LO, I_HI) for one pollutant.
# Sub-index is computed by linear interpolation inside the band that
# contains the measured concentration.
CPCB_BREAKPOINTS = {
    "PM2.5": [
        (0,   30,   0,  50),
        (31,  60,  51, 100),
        (61,  90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 1000, 401, 500),
    ],
    "PM10": [
        (0,    50,   0,  50),
        (51,  100,  51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 1000, 401, 500),
    ],
    "NO2": [
        (0,    40,   0,  50),
        (41,   80,  51, 100),
        (81,  180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    "SO2": [
        (0,    40,   0,  50),
        (41,   80,  51, 100),
        (81,  380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 3000, 401, 500),
    ],
    "O3": [
        (0,    50,   0,  50),
        (51,  100,  51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1500, 401, 500),
    ],
    "CO": [
        (0,    1.0,   0,  50),
        (1.1,  2.0,  51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 100.0, 401, 500),
    ],
}

AQI_CATEGORIES = [
    (0,    50,  "Good",                "#009966", "Air quality is satisfactory. Air pollution poses little or no risk."),
    (51,  100,  "Satisfactory",        "#84cf33", "Air quality is acceptable. Unusually sensitive people may experience minor respiratory symptoms."),
    (101, 200,  "Moderately polluted", "#ffde33", "May cause minor breathing discomfort to sensitive groups. Reduce prolonged outdoor exertion."),
    (201, 300,  "Poor",                "#ff9933", "Breathing discomfort likely for most people on prolonged exposure. Limit outdoor activity."),
    (301, 400,  "Very poor",           "#cc0033", "Respiratory effects likely for the general population. Avoid outdoor activity."),
    (401, 500,  "Severe",              "#660099", "Serious health impacts. Remain indoors and keep activity levels low."),
]

# -------------------------------------------------------------- helper funcs

@st.cache_resource(show_spinner="Loading SARIMA models…")
def load_models():
    out = {}
    for name, meta in POLLUTANTS.items():
        with open(MODEL_DIR / meta["file"], "rb") as f:
            out[name] = pickle.load(f)
    return out


def sub_index(pollutant: str, concentration: float) -> float | None:
    """CPCB sub-index for a single pollutant. Returns None if out of table."""
    if pollutant not in CPCB_BREAKPOINTS or concentration < 0:
        return None
    for c_lo, c_hi, i_lo, i_hi in CPCB_BREAKPOINTS[pollutant]:
        if c_lo <= concentration <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
    # Above the highest band: clamp to max sub-index
    return 500.0


def aqi_category(aqi: float):
    for lo, hi, name, color, advice in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return name, color, advice
    return "Severe", "#660099", AQI_CATEGORIES[-1][4]


def render_aqi_banner(aqi: float):
    name, color, advice = aqi_category(aqi)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, {color}dd, {color}77);
            padding: 1.6rem 2rem;
            border-radius: 10px;
            border: 2px solid {color};
            margin: 0.5rem 0 1rem 0;
            color: white;
        ">
            <div style="font-size:0.9rem; letter-spacing:0.1em; opacity:0.85; text-transform:uppercase;">
                Overall AQI
            </div>
            <div style="font-size:3.2rem; font-weight:700; line-height:1.1; margin:0.2rem 0;">
                {round(aqi)}
            </div>
            <div style="font-size:1.25rem; font-weight:600;">{name}</div>
            <div style="font-size:0.95rem; opacity:0.95; margin-top:0.4rem;">{advice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("🌫️ AQI Chennai")
    st.caption("SARIMA-based pollutant forecasting + CPCB India AQI calculator")

    st.markdown("### About")
    st.write(
        "Seven independent **SARIMA** models, each trained on monthly "
        "Chennai observations from the Kaggle *Air Quality in India* "
        "dataset, feed concentration forecasts into the **CPCB India "
        "sub-index formula** to produce an overall AQI."
    )

    st.markdown("### Sub-index formula")
    st.latex(r"I_p = \frac{I_{HI} - I_{LO}}{BP_{HI} - BP_{LO}} (C_p - BP_{LO}) + I_{LO}")
    st.caption("Overall AQI = max(Iₚ) across pollutants")

    st.markdown("### Training data")
    st.write("- Monthly aggregates, Jan 2015 onward")
    st.write("- AQI & O3: ~65 months; other pollutants: ~41 months")
    st.caption("Forecasts beyond training range rely on the SARIMA seasonal extension — "
               "treat far-future predictions as indicative, not exact.")


# ------------------------------------------------------------------- header

st.title("AQI Prediction & Forecasting — Chennai")
st.write(
    "Forecast pollutant concentrations for a future month, or plug in "
    "observed values directly to compute the AQI. Both flows use the "
    "Indian CPCB sub-index formula."
)

tab_forecast, tab_calculator = st.tabs(["📈 Forecast by Month", "🧮 AQI Calculator"])


# ================================================================== forecast

with tab_forecast:
    st.subheader("Pick a month to forecast")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        target_date = st.date_input(
            "Target month",
            value=date(2025, 6, 1),
            min_value=date(2018, 6, 1),
            max_value=date(2030, 12, 1),
            help="SARIMA returns month-start forecasts. Any day in the chosen month works.",
        )

    # Snap to month start
    target_month = pd.Timestamp(target_date).to_period("M").to_timestamp()

    models = load_models()

    rows = []
    for name, meta in POLLUTANTS.items():
        try:
            pred = models[name].get_prediction(
                start=target_month, end=target_month, dynamic=False
            )
            value = float(pred.predicted_mean.iloc[0])
        except Exception as e:
            value = float("nan")
            st.warning(f"{name}: forecast failed ({e})")
            continue

        if name == "AQI":
            rows.append({"pollutant": name, "value": value, "unit": meta["unit"], "sub_index": None})
        else:
            idx = sub_index(name, value)
            rows.append({"pollutant": name, "value": value, "unit": meta["unit"], "sub_index": idx})

    df = pd.DataFrame(rows)

    # Overall AQI = max sub-index across the 6 pollutants (CPCB rule).
    # The stand-alone AQI SARIMA model is kept as a cross-check.
    sub_indices = df[df["pollutant"] != "AQI"]["sub_index"].dropna()
    overall_aqi_cpcb = float(sub_indices.max()) if len(sub_indices) else float("nan")
    overall_aqi_sarima = float(df[df["pollutant"] == "AQI"]["value"].iloc[0])

    render_aqi_banner(overall_aqi_cpcb)

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "AQI (CPCB rule, max of sub-indices)",
            f"{overall_aqi_cpcb:.1f}",
            help="Computed from the six individual pollutant SARIMA forecasts via the CPCB sub-index formula.",
        )
    with c2:
        st.metric(
            "AQI (direct SARIMA forecast)",
            f"{overall_aqi_sarima:.1f}",
            delta=f"{overall_aqi_sarima - overall_aqi_cpcb:+.1f} vs. CPCB",
            help="The stand-alone AQI SARIMA model. Divergence from the CPCB value hints at "
                 "model uncertainty or non-stationarity in the AQI series.",
        )

    st.subheader(f"Per-pollutant forecast — {target_month.strftime('%B %Y')}")

    display = df.copy()
    display["value"] = display["value"].round(2)
    display["sub_index"] = display["sub_index"].round(1)
    display = display.rename(
        columns={
            "pollutant": "Pollutant",
            "value": "Forecast concentration",
            "unit": "Unit",
            "sub_index": "CPCB sub-index",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    csv = display.to_csv(index=False).encode()
    st.download_button(
        "Download forecast as CSV",
        csv,
        f"aqi_forecast_{target_month.strftime('%Y_%m')}.csv",
        "text/csv",
    )


# ================================================================ calculator

with tab_calculator:
    st.subheader("Enter observed pollutant concentrations")
    st.caption("Leave a field empty to skip that pollutant. Overall AQI is the maximum sub-index across the ones you enter.")

    col_defaults = {
        "PM2.5": 45.0, "PM10": 80.0, "NO2": 30.0,
        "SO2": 15.0, "O3": 60.0, "CO": 1.2,
    }

    cols = st.columns(3)
    entered = {}
    for i, (name, default) in enumerate(col_defaults.items()):
        with cols[i % 3]:
            val = st.number_input(
                f"{name} ({'mg/m³' if name == 'CO' else 'µg/m³'})",
                min_value=0.0,
                value=default,
                step=1.0 if name != "CO" else 0.1,
                key=f"calc_{name}",
            )
            entered[name] = val

    rows = []
    max_sub = 0.0
    dominant = None
    for name, value in entered.items():
        idx = sub_index(name, value)
        if idx is None:
            continue
        rows.append({"Pollutant": name, "Concentration": value,
                     "Unit": "mg/m³" if name == "CO" else "µg/m³",
                     "Sub-index": round(idx, 1)})
        if idx > max_sub:
            max_sub = idx
            dominant = name

    render_aqi_banner(max_sub)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Overall AQI", f"{max_sub:.1f}")
    with c2:
        st.metric("Dominant pollutant", dominant or "—",
                  help="Pollutant whose sub-index drives the overall AQI (the max).")

    st.subheader("Per-pollutant sub-indices")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("CPCB India breakpoint table"):
        bp_rows = []
        for poll, bands in CPCB_BREAKPOINTS.items():
            for c_lo, c_hi, i_lo, i_hi in bands:
                bp_rows.append({
                    "Pollutant": poll,
                    "Concentration range": f"{c_lo} – {c_hi}",
                    "Sub-index range": f"{i_lo} – {i_hi}",
                })
        st.dataframe(pd.DataFrame(bp_rows), use_container_width=True, hide_index=True)
