"""
Crime Hotspot Prediction — Interactive Dashboard
Run locally with:  streamlit run streamlit_app.py

Expects three files in ./app_data/ (produced by export_for_app.py):
  predictions.csv
  lsoa_boundaries.geojson
  model_metrics.json
"""
import json
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="London Crime Hotspot Predictor", layout="wide")

# ── Load data (cached so it only happens once per session) ──────
@st.cache_data
def load_data():
    preds = pd.read_csv("app_data/predictions.csv")
    geo = gpd.read_file("app_data/lsoa_boundaries.geojson")
    with open("app_data/model_metrics.json") as f:
        metrics = json.load(f)
    return preds, geo, metrics

preds, geo, metrics = load_data()

# ── Sidebar: model info + controls ───────────────────────────────
st.sidebar.title("Model info")
st.sidebar.markdown(f"**Best model:** {metrics['best_model']}")
st.sidebar.markdown(f"**Test period:** {metrics['test_period']}")
st.sidebar.markdown("*(predictions below are genuine one-month-ahead "
                     "out-of-sample forecasts, not values the model was trained on)*")

st.sidebar.markdown("### Performance")
best_metrics = metrics["results"][metrics["best_model"]]
st.sidebar.markdown(
    f"- F1: **{best_metrics['F1']:.3f}**\n"
    f"- Precision: **{best_metrics['Precision']:.3f}**\n"
    f"- Recall: **{best_metrics['Recall']:.3f}**\n"
    f"- ROC-AUC: **{best_metrics['ROC-AUC']:.3f}**\n"
    f"- PAI: **{best_metrics['PAI']:.2f}×** better than random"
)

st.sidebar.markdown("### Top predictive features")
for feat, val in metrics["top_features"].items():
    st.sidebar.markdown(f"- {feat}: {val:.3f}")

st.sidebar.divider()
months = sorted(preds["year_month"].unique())
month = st.sidebar.selectbox("Month", months, index=len(months) // 2)

lsoas_this_month = sorted(preds.loc[preds["year_month"] == month, "lsoa_code"].unique())
lsoa = st.sidebar.selectbox("LSOA code (optional — for detail panel)",
                             ["(none selected)"] + lsoas_this_month)

# ── Main title ─────────────────────────────────────────────────
st.title("London Crime Hotspot Prediction")
st.caption("Predicted probability that each LSOA is a crime hotspot "
           "(top 20% of crime density) in the selected month.")

# ── Prepare month's data ─────────────────────────────────────────
month_data = preds[preds["year_month"] == month].copy()
map_df = geo.merge(month_data, on="lsoa_code", how="left")

col1, col2 = st.columns([2, 1])

with col1:
    fig = px.choropleth_mapbox(
        map_df,
        geojson=map_df.geometry.__geo_interface__,
        locations=map_df.index,
        color="pred_prob",
        color_continuous_scale="YlOrRd",
        range_color=(0, 1),
        mapbox_style="carto-positron",
        center={"lat": 51.5074, "lon": -0.1278},
        zoom=9,
        opacity=0.75,
        hover_data={"lsoa_code": True, "pred_prob": ":.3f",
                     "is_hotspot": True, "total_crimes": True},
        labels={"pred_prob": "Predicted risk"},
        height=650,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(f"Month summary — {month}")
    n_actual = int(month_data["is_hotspot"].sum())
    n_pred = int(month_data["pred_binary"].sum())
    st.metric("Actual hotspots", n_actual)
    st.metric("Predicted hotspots", n_pred)

    correct = ((month_data["is_hotspot"] == 1) & (month_data["pred_binary"] == 1)).sum()
    st.metric("Correctly identified", f"{correct} / {n_actual}" if n_actual else "—")

    st.divider()
    if lsoa != "(none selected)":
        row = month_data[month_data["lsoa_code"] == lsoa].iloc[0]
        st.subheader(f"LSOA {lsoa}")
        st.metric("Predicted risk", f"{row['pred_prob']:.1%}")
        st.metric("Actually a hotspot?", "Yes" if row["is_hotspot"] == 1 else "No")
        st.metric("Total recorded crimes", int(row["total_crimes"]))
    else:
        st.info("Pick an LSOA in the sidebar to see its detailed prediction.")

st.divider()
st.caption(
    "Note: predictions use only information available before the target month "
    "(lagged crime counts and lagged spatial density). No same-month data is "
    "used as a predictor, so these figures reflect genuine forecasting "
    "performance rather than in-sample fit."
)
