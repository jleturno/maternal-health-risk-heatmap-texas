import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIG + FILE PATHS
# =========================================================
st.set_page_config(
    page_title="Texas Maternal Health Risk Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "tx_dashboard_data.csv"
GEOJSON_FILE = BASE_DIR / "data" / "tx_counties.geojson"


# =========================================================
# CUSTOM STYLE
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.35rem;
        padding-bottom: 1.4rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1.03rem;
        color: rgba(250,250,250,0.78);
        margin-bottom: 0.85rem;
        max-width: 1080px;
        line-height: 1.5;
    }

    .hero-takeaway {
        font-size: 0.98rem;
        line-height: 1.45;
        color: rgba(250,250,250,0.90);
        background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.028));
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }

    .pill {
        display: inline-block;
        padding: 0.32rem 0.74rem;
        border-radius: 999px;
        font-size: 0.79rem;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.055);
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 17px;
        padding: 0.9rem 0.95rem;
        height: 100%;
    }

    .metric-label {
        font-size: 0.78rem;
        color: rgba(250,250,250,0.68);
        margin-bottom: 0.16rem;
    }

    .metric-value {
        font-size: 1.58rem;
        font-weight: 780;
        line-height: 1.05;
    }

    .metric-sub {
        font-size: 0.80rem;
        color: rgba(250,250,250,0.66);
        margin-top: 0.32rem;
        line-height: 1.42;
    }

    .callout-card {
        background: rgba(255,255,255,0.032);
        border: 1px solid rgba(255,255,255,0.075);
        border-radius: 17px;
        padding: 0.9rem 0.95rem;
        height: 100%;
    }

    .callout-title {
        font-size: 0.76rem;
        font-weight: 750;
        color: rgba(250,250,250,0.68);
        margin-bottom: 0.32rem;
        text-transform: uppercase;
        letter-spacing: 0.035em;
    }

    .callout-text {
        font-size: 0.95rem;
        line-height: 1.45;
    }

    .small-note {
        color: rgba(250,250,250,0.68);
        font-size: 0.86rem;
        margin-bottom: 0.55rem;
        line-height: 1.45;
    }

    .footer-note {
        color: rgba(250,250,250,0.62);
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .section-caption {
        color: rgba(250,250,250,0.72);
        font-size: 0.9rem;
        line-height: 1.45;
        margin-bottom: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONSTANTS
# =========================================================
LABELS = {
    "maternal_risk_score_pct": "Composite Maternal Risk Score",
    "maternal_mortality_rate_pct": "Maternal Mortality Burden",
    "prenatal_care_access_percent_pct": "Low Prenatal Care Access",
    "median_income_pct": "Economic Risk / Low Income",
    "obesity_rate_percent_pct": "Obesity Burden",
}

METRIC_DEFINITIONS = {
    "maternal_risk_score_pct": {
        "short": "Composite percentile summarizing maternal-health-related burden across selected indicators.",
        "higher_is_worse": True,
    },
    "maternal_mortality_rate_pct": {
        "short": "Relative statewide burden of maternal mortality-related risk.",
        "higher_is_worse": True,
    },
    "prenatal_care_access_percent_pct": {
        "short": "Higher percentile indicates relatively worse prenatal care access compared with other Texas counties.",
        "higher_is_worse": True,
    },
    "median_income_pct": {
        "short": "Economic vulnerability proxy, framed so higher percentile reflects greater relative risk.",
        "higher_is_worse": True,
    },
    "obesity_rate_percent_pct": {
        "short": "Relative obesity burden, included as a maternal health-related risk context factor.",
        "higher_is_worse": True,
    },
}

MAP_COLOR_SCALE = [
    [0.00, "#f7fbff"],
    [0.15, "#e3eef8"],
    [0.35, "#c5dbef"],
    [0.55, "#90badc"],
    [0.75, "#4f8ec1"],
    [1.00, "#0f4c81"],
]


# =========================================================
# HELPERS
# =========================================================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def fmt_num(x, nd=1):
    if pd.isna(x):
        return "—"
    return f"{float(x):.{nd}f}"


def fmt_int(x):
    if pd.isna(x):
        return "—"
    return f"{int(round(float(x)))}"


def rank_suffix(n):
    if pd.isna(n):
        return ""
    n = int(n)
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def risk_tier(pct):
    if pd.isna(pct):
        return "Unknown"
    if pct >= 90:
        return "Extreme"
    if pct >= 75:
        return "High"
    if pct >= 50:
        return "Moderate"
    if pct >= 25:
        return "Low"
    return "Very Low"


def tier_description(pct):
    if pd.isna(pct):
        return "Insufficient data available for tier classification."
    if pct >= 90:
        return "County falls in the highest statewide risk group."
    if pct >= 75:
        return "County is materially above the statewide distribution."
    if pct >= 50:
        return "County is in the mid-to-upper statewide range."
    if pct >= 25:
        return "County is below the statewide midpoint."
    return "County is among the lower-risk counties statewide."


def percentile_band_text(val):
    if pd.isna(val):
        return "Percentile unavailable"
    if val >= 90:
        return "top 10% statewide"
    if val >= 75:
        return "top 25% statewide"
    if val >= 50:
        return "upper half statewide"
    if val >= 25:
        return "lower half statewide"
    return "lowest quartile statewide"


def delta_text(delta):
    if pd.isna(delta):
        return "Comparison unavailable"
    if delta > 0:
        return f"{delta:+.1f} points above Texas average"
    if delta < 0:
        return f"{delta:+.1f} points below Texas average"
    return "Aligned with Texas average"


def compute_rank(series, value):
    s = pd.Series(series).dropna().sort_values(ascending=False).reset_index(drop=True)
    matches = np.where(np.isclose(s.values, value, equal_nan=False))[0]
    if len(matches) == 0:
        return np.nan
    return int(matches[0] + 1)


def build_driver_table(row, available_layers, labels, tx_avg):
    rows = []
    for col in available_layers:
        county_val = safe_float(row.get(col, np.nan))
        tx_val = safe_float(tx_avg.get(col, np.nan))
        delta = county_val - tx_val if np.isfinite(county_val) and np.isfinite(tx_val) else np.nan
        rows.append(
            {
                "Metric Key": col,
                "Metric": labels[col],
                "County": county_val,
                "Texas Avg": tx_val,
                "Delta": delta,
                "Abs Delta": abs(delta) if not pd.isna(delta) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("Abs Delta", ascending=False)


def build_profile_chart_df(row, available_layers, labels, tx_avg):
    rows = []
    for col in available_layers:
        rows.append(
            {
                "Metric": labels[col],
                "County": safe_float(row.get(col, np.nan)),
                "Texas Avg": safe_float(tx_avg.get(col, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def find_similar_counties(df, available_layers, selected_geoid, top_n=5):
    feature_cols = [c for c in available_layers if c in df.columns]
    comp = df[["GEOID", "NAME"] + feature_cols].copy().dropna(subset=feature_cols)

    if selected_geoid not in comp["GEOID"].values:
        return pd.DataFrame()

    X = comp[feature_cols].astype(float)
    X_std = (X - X.mean()) / X.std(ddof=0).replace(0, 1)

    target_idx = comp.index[comp["GEOID"] == selected_geoid][0]
    target_vec = X_std.loc[target_idx].values
    dist = np.sqrt(((X_std.values - target_vec) ** 2).sum(axis=1))

    comp = comp.copy()
    comp["Distance"] = dist
    comp = comp[comp["GEOID"] != selected_geoid].sort_values("Distance", ascending=True).head(top_n)

    if "maternal_risk_score_pct" in comp.columns:
        comp["Composite Risk"] = comp["maternal_risk_score_pct"].round(1)

    output_cols = ["NAME", "GEOID", "Distance"]
    if "Composite Risk" in comp.columns:
        output_cols.append("Composite Risk")

    return comp[output_cols]


def make_county_summary(row, selected_layer, drivers_df, df, labels, tx_avg, total_counties):
    county_name = row["NAME"]
    geoid = row["GEOID"]

    overall = safe_float(row.get("maternal_risk_score_pct", np.nan))
    overall_rank = compute_rank(df["maternal_risk_score_pct"], overall) if "maternal_risk_score_pct" in df.columns else np.nan

    selected_val = safe_float(row.get(selected_layer, np.nan))
    tx_val = safe_float(tx_avg.get(selected_layer, np.nan))
    delta = selected_val - tx_val if np.isfinite(selected_val) and np.isfinite(tx_val) else np.nan

    top3 = drivers_df.head(3).copy()
    driver_lines = []
    for _, r in top3.iterrows():
        direction = "above" if r["Delta"] > 0 else "below"
        driver_lines.append(
            f"- {r['Metric']}: {fmt_num(r['County'])} vs Texas average {fmt_num(r['Texas Avg'])} "
            f"({fmt_num(abs(r['Delta']))} points {direction})"
        )

    report = f"""
Texas Maternal Health Risk Prioritization Snapshot
County: {county_name} County
GEOID: {geoid}

Overall Composite Risk
- Composite percentile: {fmt_num(overall)}
- Risk tier: {risk_tier(overall)}
- Statewide rank: {fmt_int(overall_rank)} of {total_counties}

Selected Layer
- Metric: {labels[selected_layer]}
- County percentile: {fmt_num(selected_val)}
- Texas average percentile: {fmt_num(tx_val)}
- Difference from Texas average: {fmt_num(delta)}

Top Drivers Relative to Texas Average
{chr(10).join(driver_lines) if driver_lines else "- Driver comparison unavailable"}

Interpretation
- Higher percentile values indicate relatively higher burden or risk compared with other Texas counties.
- This dashboard is intended as a county-level prioritization and communication tool rather than a causal model.
- County-level patterns should be interpreted alongside data source notes, timing, and local context.
    """.strip()

    return report


def section_container():
    """Create a Streamlit bordered container to avoid fragile open/close HTML wrappers."""
    return st.container(border=True)


# =========================================================
# DATA LOAD
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_CSV.exists():
        st.error(f"Missing CSV file: {DATA_CSV}")
        st.stop()

    if not GEOJSON_FILE.exists():
        st.error(f"Missing GeoJSON file: {GEOJSON_FILE}")
        st.stop()

    df = pd.read_csv(DATA_CSV, dtype={"GEOID": str})

    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    df["GEOID"] = df["GEOID"].astype(str).str.zfill(5)
    df["NAME"] = df["NAME"].astype(str).str.strip()

    for col in LABELS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, geojson


@st.cache_data(show_spinner=False)
def get_feature_lookup(geojson_obj):
    lookup = {}
    for feature in geojson_obj.get("features", []):
        props = feature.get("properties", {})
        geoid = str(props.get("GEOID", "")).zfill(5)
        lookup[geoid] = feature
    return lookup


df, geojson = load_data()
feature_lookup = get_feature_lookup(geojson)


# =========================================================
# VALIDATION
# =========================================================
required = ["GEOID", "NAME"]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    st.error(f"Missing required columns in tx_dashboard_data.csv: {missing_required}")
    st.stop()

available_layers = [k for k in LABELS if k in df.columns]
if not available_layers:
    st.error("No expected percentile columns found in tx_dashboard_data.csv.")
    st.stop()

TX_AVG = {col: float(df[col].mean()) for col in available_layers}
TOTAL_COUNTIES = int(df["NAME"].nunique())


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Texas Maternal Health")
st.sidebar.caption("Explore county-level relative risk, rankings, and driver patterns.")

selected_layer = st.sidebar.selectbox(
    "Map layer",
    available_layers,
    format_func=lambda k: LABELS[k],
)

search = st.sidebar.text_input(
    "Search county",
    value="",
    placeholder="Type a county name",
).strip().lower()

county_names = sorted(df["NAME"].dropna().unique().tolist())
filtered_names = [n for n in county_names if search in n.lower()] if search else county_names
if not filtered_names:
    filtered_names = county_names

selected_county_name = st.sidebar.selectbox("County", filtered_names)

st.sidebar.divider()
st.sidebar.subheader("Display options")
show_education = st.sidebar.toggle("Context panel", value=True)
show_methodology = st.sidebar.toggle("Methodology", value=True)
show_similar = st.sidebar.toggle("Similar counties", value=True)
show_rankings = st.sidebar.toggle("Rankings", value=True)
show_report = st.sidebar.toggle("Export summary", value=True)

st.sidebar.divider()
st.sidebar.caption(
    "Percentile values are relative within Texas counties. Higher values indicate higher estimated burden or risk."
)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-title">Texas Maternal Health Risk Prioritization Dashboard</div>
    <div class="hero-subtitle">
        County-level decision support for identifying maternal health burden, contextualizing key drivers,
        and communicating where elevated risk may warrant deeper public health review.
    </div>
    <div class="hero-takeaway">
        <b>Key takeaway:</b> Counties with elevated composite maternal risk may warrant deeper review of prenatal access,
        economic vulnerability, mortality burden, and chronic disease-related risk factors.
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This dashboard is designed for exploratory analysis and prioritization. "
    "It highlights relative maternal health burden across Texas counties and is not intended as a causal model."
)

if show_education:
    with section_container():
        st.subheader("Why this matters")
        e1, e2, e3 = st.columns(3, gap="large")

        with e1:
            st.markdown(
                """
                <div class="callout-card">
                    <div class="callout-title">Maternal health context</div>
                    <div class="callout-text">
                    Maternal health is shaped by access to care, chronic disease burden, social and economic conditions,
                    and local system capacity. County-level variation can help identify where burden may be concentrated.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with e2:
            st.markdown(
                """
                <div class="callout-card">
                    <div class="callout-title">How to read this dashboard</div>
                    <div class="callout-text">
                    Metrics are shown as percentile-style values. Higher percentiles indicate relatively higher burden
                    or risk compared with other Texas counties. This is a prioritization lens, not a causal diagnosis.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with e3:
            st.markdown(
                """
                <div class="callout-card">
                    <div class="callout-title">Practical use</div>
                    <div class="callout-text">
                    The dashboard can support screening, stakeholder communication, county comparison, and targeted
                    follow-up analysis by public health, health equity, nonprofit, or hospital strategy teams.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

if show_methodology:
    with st.expander("Methodology, metric definitions, and limitations", expanded=False):
        st.markdown(
            """
            **Methodology**
            - County-level indicators are represented as percentile-style measures within the Texas distribution.
            - Higher percentile values are interpreted as relatively higher burden or risk.
            - The composite score should be treated as a screening and prioritization tool rather than a causal model.

            **Metric definitions**
            """
        )

        defs = []
        for key in available_layers:
            defs.append(
                {
                    "Metric": LABELS[key],
                    "Definition": METRIC_DEFINITIONS.get(key, {}).get("short", "Definition not provided."),
                    "Higher = Worse": "Yes" if METRIC_DEFINITIONS.get(key, {}).get("higher_is_worse", True) else "No",
                }
            )

        st.dataframe(pd.DataFrame(defs), use_container_width=True, hide_index=True)

        st.markdown(
            """
            **Important limitations**
            - County-level summaries can mask within-county disparities.
            - Percentiles show relative statewide position, not absolute burden alone.
            - Source timing may differ across underlying indicators.
            - Results should be complemented with source documentation, trend context, and local subject-matter knowledge.
            """
        )


# =========================================================
# SELECTED COUNTY
# =========================================================
row = df.loc[df["NAME"] == selected_county_name].iloc[0]
selected_geoid = str(row["GEOID"]).zfill(5)

drivers_df = build_driver_table(row, available_layers, LABELS, TX_AVG)
profile_df = build_profile_chart_df(row, available_layers, LABELS, TX_AVG)

overall_risk = safe_float(row.get("maternal_risk_score_pct", np.nan))
overall_rank = compute_rank(df["maternal_risk_score_pct"], overall_risk) if "maternal_risk_score_pct" in df.columns else np.nan

selected_metric_val = safe_float(row.get(selected_layer, np.nan))
selected_tx_avg = safe_float(TX_AVG.get(selected_layer, np.nan))
selected_delta = selected_metric_val - selected_tx_avg if np.isfinite(selected_metric_val) and np.isfinite(selected_tx_avg) else np.nan


# =========================================================
# MAP
# =========================================================
def make_map(dataframe, geojson_obj, layer_key, selected_geoid_value, feature_map):
    fig = px.choropleth_mapbox(
        dataframe,
        geojson=geojson_obj,
        locations="GEOID",
        featureidkey="properties.GEOID",
        color=layer_key,
        color_continuous_scale=MAP_COLOR_SCALE,
        range_color=(0, 100),
        mapbox_style="carto-darkmatter",
        center={"lat": 31.0, "lon": -99.0},
        zoom=4.6,
        opacity=0.96,
        hover_name="NAME",
        hover_data={"GEOID": True, layer_key: ":.1f"},
    )

    fig.update_traces(
        marker_line_width=0.65,
        marker_line_color="rgba(255,255,255,0.20)",
        customdata=np.stack([dataframe["NAME"]], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]} County</b><br>"
            "GEOID: %{location}<br>"
            + LABELS[layer_key]
            + ": %{z:.1f}<extra></extra>"
        ),
    )

    selected_feature = feature_map.get(selected_geoid_value)
    if selected_feature is not None:
        fig.add_trace(
            go.Choroplethmapbox(
                geojson={"type": "FeatureCollection", "features": [selected_feature]},
                locations=[selected_geoid_value],
                z=[1],
                featureidkey="properties.GEOID",
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker_opacity=1.0,
                marker_line_width=3.0,
                marker_line_color="white",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        mapbox=dict(style="carto-darkmatter"),
        coloraxis_colorbar=dict(
            title=LABELS[layer_key],
            tickformat=".0f",
            thickness=14,
            len=0.74,
            x=0.97,
            y=0.50,
        ),
    )
    return fig


# =========================================================
# MAIN LAYOUT
# =========================================================
left, right = st.columns([1.65, 1.0], gap="large")

with left:
    with section_container():
        st.subheader(f"County Risk Map · {LABELS[selected_layer]}")
        st.markdown(
            '<div class="small-note">Selected county is outlined in white. Use the sidebar to change the county and metric layer.</div>',
            unsafe_allow_html=True,
        )

        map_fig = make_map(df, geojson, selected_layer, selected_geoid, feature_lookup)
        st.plotly_chart(
            map_fig,
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
        )

with right:
    with section_container():
        st.subheader(f"County Profile · {row['NAME']} County")

        st.markdown(
            f"""
            <div>
                <span class="pill">Risk Tier: {risk_tier(overall_risk)}</span>
                <span class="pill">GEOID: {row["GEOID"]}</span>
                <span class="pill">{percentile_band_text(overall_risk)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Composite Risk</div>
                    <div class="metric-value">{fmt_num(overall_risk)}</div>
                    <div class="metric-sub">{tier_description(overall_risk)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            rank_text = f"{fmt_int(overall_rank)}{rank_suffix(overall_rank)} of {TOTAL_COUNTIES}" if not pd.isna(overall_rank) else "—"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Statewide Rank</div>
                    <div class="metric-value">{rank_text}</div>
                    <div class="metric-sub">Higher rank indicates higher relative burden.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Selected Layer</div>
                    <div class="metric-value">{fmt_num(selected_metric_val)}</div>
                    <div class="metric-sub">{delta_text(selected_delta)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="callout-card">
                    <div class="callout-title">Burden level</div>
                    <div class="callout-text">
                    {row["NAME"]} County's composite maternal risk falls in the <b>{percentile_band_text(overall_risk)}</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            top_driver_name = drivers_df.iloc[0]["Metric"] if len(drivers_df) else "Unavailable"
            st.markdown(
                f"""
                <div class="callout-card">
                    <div class="callout-title">Main pressure point</div>
                    <div class="callout-text">
                    The largest deviation from the Texas average is in <b>{top_driver_name}</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                """
                <div class="callout-card">
                    <div class="callout-title">Planning implication</div>
                    <div class="callout-text">
                    Elevated percentile burden may warrant local validation and targeted maternal health planning.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown("**Selected metric**")
        st.write(
            f"{LABELS[selected_layer]}  \n"
            f"County percentile: **{fmt_num(selected_metric_val)}**  \n"
            f"Texas average: **{fmt_num(selected_tx_avg)}**  \n"
            f"Difference: **{fmt_num(selected_delta)}**"
        )

        st.markdown("**Top differences relative to Texas average**")
        drivers_display = drivers_df[["Metric", "County", "Texas Avg", "Delta"]].copy()
        drivers_display["County"] = drivers_display["County"].map(lambda x: fmt_num(x))
        drivers_display["Texas Avg"] = drivers_display["Texas Avg"].map(lambda x: fmt_num(x))
        drivers_display["Delta"] = drivers_display["Delta"].map(lambda x: fmt_num(x))
        st.dataframe(drivers_display.head(5), use_container_width=True, hide_index=True)


# =========================================================
# PROFILE CHART + INTERPRETATION
# =========================================================
st.write("")
chart_left, chart_right = st.columns([1, 1], gap="large")

with chart_left:
    with section_container():
        st.subheader("Component Profile · County vs Texas")
        st.markdown(
            '<div class="section-caption">Compares the selected county against statewide averages across available percentile indicators.</div>',
            unsafe_allow_html=True,
        )

        chart_df_long = profile_df.melt(
            id_vars="Metric",
            value_vars=["County", "Texas Avg"],
            var_name="Series",
            value_name="Percentile",
        )

        fig_profile = px.bar(
            chart_df_long,
            x="Metric",
            y="Percentile",
            color="Series",
            barmode="group",
            hover_data={"Percentile": ":.1f"},
        )

        fig_profile.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=8, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Percentile",
            legend_title="",
        )

        fig_profile.update_xaxes(tickangle=-20)
        st.plotly_chart(fig_profile, use_container_width=True, config={"displayModeBar": False})

with chart_right:
    with section_container():
        st.subheader("Analyst Interpretation")

        top3 = drivers_df.head(3).copy()

        st.markdown("**Summary**")
        st.write(
            f"{row['NAME']} County has a composite maternal risk percentile of "
            f"**{fmt_num(overall_risk)}** and ranks **{fmt_int(overall_rank)} of {TOTAL_COUNTIES}** "
            "statewide on the composite view."
        )

        st.write(
            f"For the selected metric, **{LABELS[selected_layer]}**, the county is at "
            f"**{fmt_num(selected_metric_val)}** compared with a Texas average of "
            f"**{fmt_num(selected_tx_avg)}**."
        )

        st.markdown("**Largest deviations from statewide average**")
        if len(top3) == 0:
            st.write("Driver comparison unavailable.")
        else:
            for _, r in top3.iterrows():
                direction = "higher than" if r["Delta"] > 0 else "lower than"
                st.write(
                    f"- **{r['Metric']}**: {fmt_num(abs(r['Delta']))} points {direction} the Texas average."
                )

        st.markdown("**Interpretive note**")
        st.write(
            "Elevated percentile burden should be read as a signal for prioritization and deeper investigation, "
            "not proof of causation. Findings should be interpreted alongside local context, data source timing, "
            "and subject-matter expertise."
        )


# =========================================================
# SIMILAR COUNTIES
# =========================================================
if show_similar:
    st.write("")
    with section_container():
        st.subheader("Most Similar Counties")
        st.markdown(
            '<div class="section-caption">Similarity is based on available percentile metrics. Lower distance indicates a more similar county profile.</div>',
            unsafe_allow_html=True,
        )

        similar_df = find_similar_counties(df, available_layers, selected_geoid, top_n=5)

        if len(similar_df) == 0:
            st.info("Similar county comparison unavailable.")
        else:
            similar_show = similar_df.copy()
            if "Distance" in similar_show.columns:
                similar_show["Distance"] = similar_show["Distance"].map(lambda x: f"{x:.2f}")
            st.dataframe(similar_show, use_container_width=True, hide_index=True)


# =========================================================
# RANKINGS
# =========================================================
if show_rankings:
    st.write("")

    ranked = (
        df[["NAME", "GEOID", selected_layer]]
        .dropna()
        .sort_values(selected_layer, ascending=False)
        .reset_index(drop=True)
        .copy()
    )
    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    ranked["Percentile"] = ranked[selected_layer].map(lambda x: round(float(x), 1))

    selected_rank_row = ranked.loc[ranked["GEOID"] == selected_geoid]
    selected_rank_text = "Unavailable"
    if len(selected_rank_row):
        sr = selected_rank_row.iloc[0]
        selected_rank_text = f'{int(sr["Rank"])}{rank_suffix(int(sr["Rank"]))} of {len(ranked)}'

    left_rank, right_rank = st.columns(2, gap="large")

    with left_rank:
        with section_container():
            st.subheader(f"Highest-Burden Counties · {LABELS[selected_layer]}")
            st.dataframe(
                ranked[["Rank", "NAME", "GEOID", "Percentile"]].head(10),
                use_container_width=True,
                hide_index=True,
            )

    with right_rank:
        with section_container():
            st.subheader(f"Lowest-Burden Counties · {LABELS[selected_layer]}")
            st.dataframe(
                ranked[["Rank", "NAME", "GEOID", "Percentile"]].tail(10).sort_values("Rank", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(f"**Selected county rank in this layer:** {selected_rank_text}")


# =========================================================
# EXPORTS
# =========================================================
if show_report:
    st.write("")
    with section_container():
        st.subheader("Narrative Summary and Exports")
        st.markdown(
            '<div class="section-caption">Generate a county-level summary for policy communication, stakeholder review, or follow-up analysis.</div>',
            unsafe_allow_html=True,
        )

        report_text = make_county_summary(row, selected_layer, drivers_df, df, LABELS, TX_AVG, TOTAL_COUNTIES)
        st.text_area("County summary", value=report_text, height=245, label_visibility="collapsed")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.download_button(
                "Download county summary (.txt)",
                data=report_text.encode("utf-8"),
                file_name=f"{row['NAME'].replace(' ', '_')}_maternal_health_summary.txt",
                mime="text/plain",
            )

        with export_col2:
            export_df = drivers_df[["Metric", "County", "Texas Avg", "Delta"]].copy()
            st.download_button(
                "Download driver comparison (.csv)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{row['NAME'].replace(' ', '_')}_driver_comparison.csv",
                mime="text/csv",
            )


# =========================================================
# FOOTER
# =========================================================
st.write("")
st.markdown(
    """
    <div class="footer-note">
    Designed as a county-level maternal health prioritization and communication tool for Texas.
    Percentile-style values indicate relative statewide position and should be interpreted with source context,
    timing, and local expertise.
    </div>
    """,
    unsafe_allow_html=True,
)
