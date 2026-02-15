"""
AqarSense — Intelligent Rental Estimation for UAE Real Estate
Streamlit UI · Uses the trained Ridge pipeline via src.predict.predict_rent()
"""

import pathlib
import base64
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

from src.predict import predict_rent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "dubai_properties.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ridge_model.pkl"
BG_IMAGE_PATH = PROJECT_ROOT / "assets" / "AqarSense bg.jpg"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AqarSense",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Background Image Helper
# ---------------------------------------------------------------------------
def get_base64_image(image_path):
    """Convert image to base64 for CSS embedding."""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

bg_image_b64 = get_base64_image(BG_IMAGE_PATH)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
bg_style = ""
if bg_image_b64:
    bg_style = f"""
    [data-testid="stAppViewContainer"] > div:first-child {{
        background: linear-gradient(rgba(0, 0, 0, 0.78), rgba(0, 0, 0, 0.78)), 
                    url("data:image/jpg;base64,{bg_image_b64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    """
else:
    bg_style = """
    [data-testid="stAppViewContainer"] > div:first-child {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    """

st.markdown(
    f"""
    <style>
    /* ---------- global + background ---------- */
    {bg_style}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}

    /* ---------- header ---------- */
    .app-header {{ text-align: center; padding: 1.2rem 0 0.2rem 0; }}
    .app-header h1 {{
        font-size: 2.6rem; font-weight: 700; color: #ffffff;
        letter-spacing: 0.02em; margin-bottom: 0.15rem;
    }}
    .app-header p {{
        font-size: 1.05rem; color: #ffffff; margin-top: 0;
    }}

    /* ---------- prediction card ---------- */
    .pred-card {{
        background: #f8f9fb; border: 1px solid #dde1e7;
        border-radius: 12px; padding: 1.8rem 1.4rem;
        text-align: center; margin-bottom: 1rem;
    }}
    .pred-value {{
        font-size: 2.6rem; font-weight: 700; color: #0d3b66;
        margin: 0.4rem 0 0.15rem 0;
    }}
    .pred-range {{
        font-size: 0.95rem; color: #666; margin-bottom: 0.6rem;
    }}
    .segment-badge {{
        display: inline-block; padding: 0.3rem 1rem;
        border-radius: 20px; font-weight: 600; font-size: 0.92rem;
    }}
    .seg-budget    {{ background: #e8f5e9; color: #2e7d32; }}
    .seg-mid       {{ background: #e3f2fd; color: #1565c0; }}
    .seg-uppermid  {{ background: #fff3e0; color: #e65100; }}
    .seg-premium   {{ background: #fce4ec; color: #b71c1c; }}

    /* ---------- feature list ---------- */
    .feat-list {{ list-style: none; padding-left: 0; }}
    .feat-list li {{
        padding: 0.35rem 0; border-bottom: 1px solid #444;
        font-size: 0.93rem; color: #ffffff;
    }}
    .feat-list li:last-child {{ border-bottom: none; }}
    .feat-rank {{
        display: inline-block; width: 22px; height: 22px;
        border-radius: 50%; background: #0d3b66; color: #fff;
        text-align: center; line-height: 22px; font-size: 0.75rem;
        margin-right: 0.5rem;
    }}

    /* ---------- footer ---------- */
    .app-footer {{
        text-align: center; padding: 2rem 0 1rem 0;
        color: #999; font-size: 0.85rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading & precomputation (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def precompute_lookups():
    df = load_dataset()
    cities = sorted(df["City"].dropna().unique())
    locations_by_city = (
        df.groupby("City")["Location"]
        .apply(lambda s: sorted(s.dropna().unique()))
        .to_dict()
    )
    # Group by (City, Location) so same location name in different cities
    # resolves to the correct coordinates
    coord_means = (
        df.groupby(["City", "Location"])[["Latitude", "Longitude"]]
        .mean()
        .to_dict(orient="index")
    )
    rent_quartiles = df["Rent"].quantile([0.25, 0.50, 0.75]).to_dict()
    property_types = sorted(df["Type"].dropna().unique())
    furnishings = sorted(df["Furnishing"].dropna().unique())
    return cities, locations_by_city, coord_means, rent_quartiles, property_types, furnishings


@st.cache_resource
def load_model_pipeline():
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
AREA_RANGES = {
    "0 – 500 sqft": 250,
    "500 – 1,000 sqft": 750,
    "1,000 – 1,500 sqft": 1250,
    "1,500 – 2,000 sqft": 1750,
    "2,000 – 3,000 sqft": 2500,
    "3,000+ sqft": 4000,
}

MULT_ERROR = 1.62  # from 5-fold CV


def classify_segment(rent: float, q: dict) -> tuple:
    if rent <= q[0.25]:
        return "Budget", "seg-budget"
    elif rent <= q[0.50]:
        return "Mid-Market", "seg-mid"
    elif rent <= q[0.75]:
        return "Upper Mid-Market", "seg-uppermid"
    else:
        return "Premium", "seg-premium"


def get_top_features(pipeline, n: int = 5) -> list:
    """Extract top-n feature names by absolute Ridge coefficient."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    coefs = np.abs(model.coef_)
    feature_names = preprocessor.get_feature_names_out()
    top_idx = np.argsort(coefs)[::-1][:n]
    names = []
    for i in top_idx:
        raw = feature_names[i]
        # Clean transformer prefixes e.g. "num__Beds" -> "Beds", "cat__Type_Apartment" -> "Type: Apartment"
        if raw.startswith("num__"):
            names.append(raw[5:])
        elif raw.startswith("cat__"):
            parts = raw[5:].split("_", 1)
            names.append(f"{parts[0]}: {parts[1]}" if len(parts) == 2 else parts[0])
        else:
            names.append(raw)
    return names


def get_local_influences(pipeline, input_df: pd.DataFrame, n: int = 5) -> list:
    """Compute top-n local feature contributions (absolute) for a single input.

    Contributions are computed in log-space as coef_i * x_i for the linear Ridge model.
    Returns readable feature labels ordered by absolute contribution descending.
    """
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    Xt = pre.transform(input_df)
    try:
        x_arr = Xt.toarray().ravel()
    except Exception:
        x_arr = np.asarray(Xt).ravel()

    coefs = model.coef_
    contribs = coefs * x_arr
    feat_names = pre.get_feature_names_out()

    top_idx = np.argsort(np.abs(contribs))[::-1][:n]
    labels = []
    for i in top_idx:
        raw = feat_names[i]
        if raw.startswith("num__"):
            labels.append(raw[5:])
        elif raw.startswith("cat__"):
            parts = raw[5:].split("_", 1)
            labels.append(f"{parts[0]}: {parts[1]}" if len(parts) == 2 else parts[0])
        else:
            labels.append(raw)
    return labels


# ---------------------------------------------------------------------------
# Load data + lookups
# ---------------------------------------------------------------------------
df = load_dataset()
(
    cities,
    locations_by_city,
    coord_means,
    rent_quartiles,
    property_types,
    furnishings,
) = precompute_lookups()
pipeline = load_model_pipeline()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header">'
    "<h1>AqarSense</h1>"
    "<p>Intelligent Rental Estimation for UAE Real Estate</p>"
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# SECTION 1 — Property Input
# ---------------------------------------------------------------------------
st.subheader("Property Details")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    selected_city = st.selectbox(
        "City",
        options=[None] + cities,
        format_func=lambda x: "Select City" if x is None else x,
        index=0
    )
    available_locations = locations_by_city.get(selected_city, []) if selected_city else []
    selected_location = st.selectbox(
        "Location",
        options=[None] + available_locations,
        format_func=lambda x: "Select Location" if x is None else x,
        index=0
    )
    selected_type = st.selectbox(
        "Property Type",
        options=[None] + property_types,
        format_func=lambda x: "Select Property Type" if x is None else x,
        index=0
    )
    selected_furnishing = st.selectbox(
        "Furnishing",
        options=[None] + furnishings,
        format_func=lambda x: "Select Furnishing" if x is None else x,
        index=0
    )

with col_right:
    beds = st.number_input("Bedrooms", min_value=0, max_value=20, value=0, step=1, help="Enter number of bedrooms")
    baths = st.number_input("Bathrooms", min_value=0, max_value=20, value=0, step=1, help="Enter number of bathrooms")
    area_label = st.selectbox(
        "Area (sqft)",
        options=[None] + list(AREA_RANGES.keys()),
        format_func=lambda x: "Select Area Range" if x is None else x,
        index=0
    )
    area_sqft = AREA_RANGES.get(area_label, 0) if area_label else 0

# Derive coordinates using (City, Location) to avoid cross-city collisions
coords = coord_means.get((selected_city, selected_location), {})
derived_lat = coords.get("Latitude", np.nan)
derived_lon = coords.get("Longitude", np.nan)

st.markdown("")
# Disable predict button if required fields are not selected
all_fields_filled = all([
    selected_city, selected_location, selected_type,
    selected_furnishing, area_label, beds >= 0, baths >= 0
])
if not all_fields_filled:
    st.info("⚠️ Please fill in all property details to get a prediction.")
predict_clicked = st.button(
    "Predict Rent",
    type="primary",
    use_container_width=True,
    disabled=not all_fields_filled
)

# ---------------------------------------------------------------------------
# SECTION 2 — Prediction Output
# ---------------------------------------------------------------------------
if predict_clicked:
    input_dict = {
        "City": selected_city,
        "Location": selected_location,
        "Type": selected_type,
        "Furnishing": selected_furnishing,
        "Beds": beds,
        "Baths": baths,
        "Area_in_sqft": area_sqft,
        "Latitude": derived_lat,
        "Longitude": derived_lon,
        "Posted_date": None,
        "Age_of_listing_in_days": None,
    }

    predicted_rent = predict_rent(input_dict)
    lower_bound = predicted_rent / MULT_ERROR
    upper_bound = predicted_rent * MULT_ERROR
    segment_label, segment_cls = classify_segment(predicted_rent, rent_quartiles)
    # Compute local top feature influences (absolute contributions) for this input
    local_features = get_local_influences(pipeline, pd.DataFrame([input_dict]), n=5)

    st.divider()
    st.subheader("Prediction")

    res_left, res_right = st.columns([3, 2], gap="large")

    with res_left:
        st.markdown(
            f'<div class="pred-card">'
            f'<div style="font-size:0.9rem;color:#888;">Estimated Annual Rent</div>'
            f'<div class="pred-value">AED {predicted_rent:,.0f}</div>'
            f'<div class="pred-range">Range: AED {lower_bound:,.0f} — AED {upper_bound:,.0f}</div>'
            f'<span class="segment-badge {segment_cls}">{segment_label}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with res_right:
        st.markdown("**Top Feature Influences (local)**")
        feat_html = '<ul class="feat-list">'
        for idx, feat in enumerate(local_features, start=1):
            feat_html += f'<li><span class="feat-rank">{idx}</span>{feat}</li>'
        feat_html += "</ul>"
        st.markdown(feat_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------
    # SECTION 3 — Visualizations
    # -------------------------------------------------------------------
    st.divider()
    st.subheader("Analysis")

    # 3.1  Rent Distribution with predicted overlay
    # Clip at 99th percentile so the bulk of the distribution is visible
    st.markdown("**Rent Distribution**")
    rent_cap = df["Rent"].quantile(0.99)
    hist_df = df[df["Rent"] <= rent_cap]
    fig_hist = px.histogram(
        hist_df,
        x="Rent",
        nbins=100,
        color_discrete_sequence=["#5c8a97"],
        labels={"Rent": "Annual Rent (AED)", "count": "Properties"},
    )
    fig_hist.add_vline(
        x=min(predicted_rent, rent_cap),
        line_width=2.5,
        line_dash="dash",
        line_color="#d32f2f",
        annotation_text=f"AED {predicted_rent:,.0f}",
        annotation_position="top right",
        annotation_font_color="#d32f2f",
    )
    fig_hist.update_layout(
        template="plotly_white",
        xaxis_title="Annual Rent (AED)",
        yaxis_title="Number of Properties",
        bargap=0.05,
        height=380,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 3.2  Map tabs
    tab_loc, tab_heat = st.tabs(["📍 Property Location", "🗺️ Rent Heatmap"])

    with tab_loc:
        if not np.isnan(derived_lat) and not np.isnan(derived_lon):
            fig_loc = px.scatter_mapbox(
                lat=[derived_lat],
                lon=[derived_lon],
                zoom=14,
                size=[1],
                size_max=18,
                color_discrete_sequence=["#d32f2f"],
            )
            fig_loc.update_layout(
                mapbox_style="carto-positron",
                mapbox_center={"lat": derived_lat, "lon": derived_lon},
                height=450,
                margin=dict(t=0, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_loc, use_container_width=True)
        else:
            st.info("Coordinates not available for the selected location.")

    with tab_heat:
        heat_df = df.dropna(subset=["Latitude", "Longitude", "Rent"]).copy()
        # Sample for performance
        if len(heat_df) > 8000:
            heat_df = heat_df.sample(n=8000, random_state=42)
        # Cap color scale at 95th percentile so outliers don't wash out variation
        rent_p95 = heat_df["Rent"].quantile(0.95)
        heat_df["Rent_capped"] = heat_df["Rent"].clip(upper=rent_p95)
        fig_map = px.scatter_mapbox(
            heat_df,
            lat="Latitude",
            lon="Longitude",
            color="Rent_capped",
            color_continuous_scale="YlOrRd",
            range_color=[0, rent_p95],
            size_max=8,
            zoom=8,
            opacity=0.65,
            labels={"Rent_capped": "Rent (AED)"},
        )
        fig_map.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": derived_lat, "lon": derived_lon} if not np.isnan(derived_lat) else {},
            mapbox_zoom=10 if not np.isnan(derived_lat) else 8,
            height=500,
            margin=dict(t=10, b=10, l=0, r=0),
            coloraxis_colorbar=dict(title="Rent (AED)"),
        )
        st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="app-footer">A project by <strong>Nalan Baburajan</strong></div>',
    unsafe_allow_html=True,
)
