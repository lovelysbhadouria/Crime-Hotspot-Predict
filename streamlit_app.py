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


def classification_metrics(data):
    true_positive = int(((data["is_hotspot"] == 1) & (data["pred_binary"] == 1)).sum())
    false_positive = int(((data["is_hotspot"] == 0) & (data["pred_binary"] == 1)).sum())
    false_negative = int(((data["is_hotspot"] == 1) & (data["pred_binary"] == 0)).sum())
    true_negative = int(((data["is_hotspot"] == 0) & (data["pred_binary"] == 0)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = (true_positive + true_negative) / len(data) if len(data) else 0
    return accuracy, precision, recall, f1


exported_metrics = classification_metrics(preds)
metadata_metrics = metrics["results"][metrics["best_model"]]
metrics_differ = any(
    abs(exported_value - metadata_metrics[key]) > 0.01
    for exported_value, key in zip(exported_metrics, ["Accuracy", "Precision", "Recall", "F1"])
)

# ── Sidebar: controls ────────────────────────────────────────────
st.sidebar.title("Choose a month")
months = sorted(preds["year_month"].unique())
month = st.sidebar.selectbox("Month", months, index=0)

lsoas_this_month = sorted(preds.loc[preds["year_month"] == month, "lsoa_code"].unique())
lsoa = st.sidebar.text_input(
    "Area code (optional)",
    placeholder="e.g. E01000001",
    help="Type an area code to see its details. Leave blank to explore the map only.",
).strip().upper()
if lsoa and lsoa not in lsoas_this_month:
    st.sidebar.warning("That area code is not available for the selected month.")
    lsoa = ""
map_mode = st.sidebar.radio(
    "What to show on the map",
    ["Changes since last month", "Relative risk level", "Crimes recorded"],
    help="Changes since last month makes differences easier to see. Relative risk level groups areas by the model's ranking. Crimes recorded shows this month's reported crime count.",
)

with st.sidebar.expander("How this forecast was made"):
    st.markdown(f"**Forecast method:** {metrics['best_model']}")
    st.markdown(f"**Forecast period tested:** {metrics['test_period']}")
    st.caption("The map shows a genuine one-month-ahead forecast, using information available before the selected month.")
    st.markdown(
        f"**Forecast quality**\n\n"
        f"Overall accuracy: **{exported_metrics[0]:.1%}**  \n"
        f"Flagged areas that were hotspots: **{exported_metrics[1]:.1%}**  \n"
        f"Recorded hotspots found: **{exported_metrics[2]:.1%}**  \n"
        f"Overall balance score: **{exported_metrics[3]:.3f}**"
    )
    st.caption("Calculated from the prediction rows displayed by this app. The selected month can perform differently.")

if metrics_differ:
    st.warning(
        "The stored model report does not exactly match the exported prediction rows. "
        "The headline metrics above are recalculated from the displayed rows; investigate "
        "the original training/export pipeline before using the historical model comparison."
    )

# ── Main title ─────────────────────────────────────────────────
st.title("London Crime Hotspot Prediction")
st.caption("See which neighbourhood areas are most likely to be crime hotspots in the selected month.")
first_month = month == months[0]
if map_mode == "Changes since last month" and first_month:
    map_help = "**How to read the map:** red areas are flagged as hotspots and grey areas are not flagged. This is the first available month, so there is no previous month for comparison."
elif map_mode == "Changes since last month":
    map_help = ("**How to read the map:** red shows new or continuing hotspots, grey means the area is "
                "not currently flagged, and blue shows an area no longer flagged compared with the previous month.")
elif map_mode == "Relative risk level":
    map_help = "**How to read the map:** darker red means a higher relative risk level. This is a ranking, not a guarantee that crime will occur."
else:
    map_help = "**How to read the map:** darker blue means more crimes were recorded in the selected month."
st.info(map_help + " Hover over an area to see its neighbourhood code and prediction. "
        "A hotspot is a forecast based on recent patterns, not a promise that crimes will be recorded in that month.")

# ── Prepare month's data ─────────────────────────────────────────
month_data = preds[preds["year_month"] == month].copy()
month_position = months.index(month)
if month_position > 0:
    previous_month = months[month_position - 1]
    previous_flags = preds.loc[
        preds["year_month"] == previous_month, ["lsoa_code", "pred_binary"]
    ].rename(columns={"pred_binary": "previous_pred_binary"})
    month_data = month_data.merge(previous_flags, on="lsoa_code", how="left")
    month_data["map_status"] = "Not flagged"
    month_data.loc[
        (month_data["pred_binary"] == 1) & (month_data["previous_pred_binary"] == 0),
        "map_status",
    ] = "New hotspot"
    month_data.loc[
        (month_data["pred_binary"] == 1) & (month_data["previous_pred_binary"] == 1),
        "map_status",
    ] = "Still a hotspot"
    month_data.loc[
        (month_data["pred_binary"] == 0) & (month_data["previous_pred_binary"] == 1),
        "map_status",
    ] = "No longer flagged"
else:
    month_data["previous_pred_binary"] = pd.NA
    month_data["map_status"] = month_data["pred_binary"].map(
        {1: "Hotspot", 0: "Not flagged"}
    )
month_data["signal_position"] = "Below hotspot cutoff"
month_data["risk_score"] = month_data["pred_prob"] * 100
month_data["risk_band"] = pd.qcut(
    month_data["pred_prob"].rank(method="first"),
    3,
    labels=["Lower relative risk", "Middle relative risk", "Higher relative risk"],
)
boundary = month_data.loc[month_data["pred_binary"] == 1, "pred_prob"].min()
month_data.loc[
    (month_data["pred_prob"] - boundary).abs() < 0.1,
    "signal_position",
] = "Near hotspot cutoff"
month_data.loc[
    (month_data["pred_binary"] == 1) & ((month_data["pred_prob"] - boundary) >= 0.1),
    "signal_position",
] = "Above hotspot cutoff"
map_df = geo.merge(month_data, on="lsoa_code", how="left")
map_df = map_df.reset_index(drop=True)  # clean 0..n-1 index, used as the id link below

col1, col2 = st.columns([2, 1])

with col1:
    # IMPORTANT: pass the geometry of the FULL GeoDataFrame (not just
    # `.geometry.__geo_interface__`) so each feature carries an "id"
    # matching map_df's index. Without this, Plotly cannot reliably
    # pair each polygon to its pred_prob value and the fill color
    # becomes disconnected from the data (this was the bug causing
    # the map to look identical — and wrongly colored — every month).
    map_kwargs = dict(
        data_frame=map_df,
        geojson=map_df.__geo_interface__,
        locations=map_df.index,
        featureidkey="id",
        mapbox_style="carto-positron",
        center={"lat": 51.5074, "lon": -0.1278},
        zoom=9,
        opacity=0.75,
        hover_data={"lsoa_code": True, "risk_band": True,
                    "pred_binary": True, "total_crimes": True,
                    "map_status": True, "signal_position": True},
        labels={"risk_band": "Relative risk level",
                "pred_binary": "Flagged as hotspot",
                "total_crimes": "Crimes recorded this month",
                "map_status": "Change since last month",
                "signal_position": "Position relative to cutoff"},
        height=650,
    )
    if map_mode == "Changes since last month":
        map_kwargs.update(
            color="map_status",
            color_discrete_map={
                "New hotspot": "#d73027",
                "Still a hotspot": "#fc8d59",
                "No longer flagged": "#91bfdb",
                "Not flagged": "#e8e8e8",
                "Hotspot": "#d73027",
            },
        )
    elif map_mode == "Relative risk level":
        map_kwargs.update(
            color="risk_band",
            category_orders={
                "risk_band": [
                    "Lower relative risk",
                    "Middle relative risk",
                    "Higher relative risk",
                ]
            },
            color_discrete_map={
                "Lower relative risk": "#91bfdb",
                "Middle relative risk": "#ffffbf",
                "Higher relative risk": "#d73027",
            },
        )
    else:
        map_kwargs.update(
            color="total_crimes",
            color_continuous_scale="Blues",
            range_color=(0, max(1, int(month_data["total_crimes"].max()))),
        )
    fig = px.choropleth_mapbox(**map_kwargs)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

with col2:
    st.subheader(f"Month summary — {month}")
    n_actual = int(month_data["is_hotspot"].sum())
    n_pred = int(month_data["pred_binary"].sum())
    st.metric("Areas flagged as hotspots", f"{n_pred:,}")
    st.caption("Areas the forecast flags for attention")
    st.metric("Hotspots in recorded data", f"{n_actual:,}")

    correct = ((month_data["is_hotspot"] == 1) & (month_data["pred_binary"] == 1)).sum()
    flagged_correctly = correct / n_actual if n_actual else 0
    flagged_precision = correct / n_pred if n_pred else 0
    st.metric("Recorded hotspots correctly flagged", f"{correct:,} / {n_actual:,}" if n_actual else "—")
    st.caption(
        f"Found **{flagged_correctly:.1%}** of recorded hotspots; "
        f"**{flagged_precision:.1%}** of flagged areas were hotspots."
    )

    if month_position > 0:
        baseline_correct = (
            (month_data["is_hotspot"] == 1)
            & (month_data["previous_pred_binary"] == 1)
        ).sum()
        new_count = int((month_data["map_status"] == "New hotspot").sum())
        dropped_count = int((month_data["map_status"] == "No longer flagged").sum())
        st.caption(
            f"Compared with {previous_month}: **{new_count:,}** new, "
            f"**{dropped_count:,}** no longer flagged."
        )
        st.metric("Previous month baseline", f"{baseline_correct:,} / {n_actual:,}" if n_actual else "—")

    st.divider()
    if lsoa:
        row = month_data[month_data["lsoa_code"] == lsoa].iloc[0]
        st.subheader(f"Selected area: {lsoa}")
        st.metric("Relative risk level", row["risk_band"])
        st.metric("Flagged as hotspot?", "Yes" if row["pred_binary"] == 1 else "No")
        st.metric("Crimes recorded this month", int(row["total_crimes"]))
        st.caption(f"Change: **{row['map_status']}** · {row['signal_position']}")
        if row["total_crimes"] == 0 and row["pred_binary"] == 1:
            st.warning(
                "No crimes are recorded for this month, but the model places this area in a higher relative-risk group "
                "based on earlier patterns. This is not a guarantee or a percentage chance of a crime. "
                "The zero count may also reflect reporting or timing differences."
            )
        else:
            st.caption("The hotspot flag is a forecast based on earlier patterns; the crime count is the number recorded for this month.")
    else:
        st.info("Enter an area code in the sidebar for a plain-language area summary.")

st.divider()
ranked = month_data.sort_values(
    ["pred_binary", "pred_prob"], ascending=[False, False]
).head(10).copy()
ranked.insert(0, "Rank", range(1, len(ranked) + 1))
ranked = ranked[["Rank", "lsoa_code", "risk_band", "total_crimes", "map_status", "signal_position"]]
ranked.columns = ["Rank", "Area", "Relative risk level", "Crimes recorded", "Change", "Position relative to cutoff"]
st.subheader("Top areas flagged for attention")
st.caption("These are the ten highest-ranked areas for the selected month. Relative risk is a ranking, not a probability or a guarantee that a crime will occur.")
st.dataframe(
    ranked,
    use_container_width=True,
    hide_index=True,
)
download_data = month_data[["lsoa_code", "year_month", "risk_score", "risk_band", "pred_binary", "is_hotspot", "total_crimes", "map_status", "signal_position"]].copy()
st.download_button(
    "Download this month's results",
    data=download_data.to_csv(index=False).encode("utf-8"),
    file_name=f"crime_hotspot_predictions_{month}.csv",
    mime="text/csv",
)

with st.expander("Model checks and limitations"):
    st.markdown(
        "**How to judge the forecast**\n\n"
        "The previous month's hotspot list is shown as a simple baseline. "
        "A useful model should match more recorded hotspots than that baseline. "
        "These results measure area-level patterns, not individual people or exact incidents."
    )
    st.markdown(
        "**Important limitations**\n\n"
        "- Recorded crime can be affected by reporting delays and differences in reporting.\n"
        "- A hotspot flag indicates elevated area-level risk; it does not prove that crime will occur.\n"
        "- The forecast should support human judgement, not replace it or determine action by itself.\n"
        "- The model does not identify people or predict individual behaviour."
    )
    st.markdown("**Comparison of forecasting methods from the stored training report**")
    model_table = pd.DataFrame(metrics["results"]).T[
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PAI"]
    ].rename_axis("Forecast method")
    st.dataframe(
        model_table.style.format({
            "Accuracy": "{:.1%}",
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "F1": "{:.3f}",
            "ROC-AUC": "{:.3f}",
            "PAI": "{:.2f}x",
        }),
        use_container_width=True,
    )
    true_positive = int(((month_data["is_hotspot"] == 1) & (month_data["pred_binary"] == 1)).sum())
    false_positive = int(((month_data["is_hotspot"] == 0) & (month_data["pred_binary"] == 1)).sum())
    false_negative = int(((month_data["is_hotspot"] == 1) & (month_data["pred_binary"] == 0)).sum())
    true_negative = int(((month_data["is_hotspot"] == 0) & (month_data["pred_binary"] == 0)).sum())
    st.caption(
        f"For {month}: **{true_positive:,}** correct hotspot flags, "
        f"**{false_positive:,}** false alarms, **{false_negative:,}** missed hotspots, "
        f"and **{true_negative:,}** correctly unflagged areas."
    )

st.caption(
    "Note: predictions use only information available before the target month "
    "(lagged crime counts and lagged spatial density). No same-month data is "
    "used as a predictor, so these figures reflect genuine forecasting "
    "performance rather than in-sample fit."
)
