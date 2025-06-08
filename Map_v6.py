import streamlit as st
import folium
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from branca.colormap import linear
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import altair as alt
from folium.raster_layers import ImageOverlay
from tempfile import NamedTemporaryFile
from folium.plugins import Draw
from rasterio.features import rasterize
from io import BytesIO
import requests

st.set_page_config(layout="wide")
sns.set_style("white")

# Google Drive download with validation and caching
@st.cache_data(show_spinner="Downloading file...")
def gdrive_download(file_id, suffix):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if not response.ok or "text/html" in response.headers.get("Content-Type", ""):
        st.error("Failed to download a valid file. Ensure it's publicly shared and not too large.")
        st.stop()
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    return tmp.name

# File IDs
WIND_FILE_ID = "1cEMvR4O4Z2m6TyLXGxccoho0T7SL9_Pg"
AIR_FILE_ID = "1IEympzffE3LZMMl1Hqs-HqGm2cQ9JokA"
RESTRICTED_FILE_ID = "1No-TTcys7wJl_GqPtTnEGTxvxVg_Tlb-"

# Download files
wind_path = gdrive_download(WIND_FILE_ID, ".tif")
air_path = gdrive_download(AIR_FILE_ID, ".tif")
restricted_path = gdrive_download(RESTRICTED_FILE_ID, ".geojson")

# Sidebar inputs
st.sidebar.header("📈 Financial Assumptions")
unlevered_beta = st.sidebar.number_input("Unlevered Beta", 0.0, 5.0, 0.8)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0)
market_risk_premium = st.sidebar.number_input("Market Risk Premium (%)", 0.0, 15.0, 5.0)
cost_of_debt = st.sidebar.slider("Cost of Debt (%)", 0.0, 15.0, 3.0, 0.1)
loan_pct = st.sidebar.slider("Debt Ratio (%)", 0, 100, 70, 1)
tax_rate = st.sidebar.slider("Corporate Tax Rate (%)", 0.0, 50.0, 25.0, 0.5)
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 2.0, 0.1)
electricity_price = st.sidebar.number_input("Electricity Price (€/MWh)", 0.0, 1000.0, 50.0)
installed_mw = st.sidebar.slider("Installed Capacity (MW)", 1, 50, 5)
om_cost = st.sidebar.number_input("O&M Cost (€/MW/year)", 0.0, 1e6, 40000.0)
st.sidebar.number_input("Air Density Normalization (kg/m³)", 1.0, 1.5, 1.225, 0.005, key="air_density_norm")

# WACC
levered_beta = unlevered_beta * (1 + (loan_pct / 100))
debt_ratio = min(max(loan_pct / 100, 0), 1)
equity_ratio = 1 - debt_ratio
cost_of_equity = risk_free_rate + levered_beta * market_risk_premium
after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate / 100)
wacc = equity_ratio * cost_of_equity + debt_ratio * after_tax_cost_of_debt
wacc = max(wacc, 0)

# Load rasters and Germany outline
def load_rasters(wind, air):
    germany = gpd.read_file("https://raw.githubusercontent.com/johan/world.geo.json/master/countries/DEU.geo.json")
    with rasterio.open(wind) as wind_src, rasterio.open(air) as air_src:
        germany = germany.to_crs(wind_src.crs)
        return wind_src.read(1), air_src.read(1), wind_src.transform, germany, wind_src.crs

wind_data, air_data, transform, germany, crs = load_rasters(wind_path, air_path)

# Clean up NaNs and infs
wind_data = np.nan_to_num(wind_data, nan=0.0, posinf=0.0, neginf=0.0)
air_data = np.nan_to_num(air_data, nan=0.0, posinf=0.0, neginf=0.0)

# Overlay image helper
def create_overlay(data):
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap("viridis")
    rgba = cmap(norm)
    rgba[..., 3] = (data > 0).astype(float)
    fig, ax = plt.subplots()
    ax.imshow(rgba, origin='upper')
    ax.axis('off')
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches='tight', pad_inches=0, dpi=150, transparent=True)
        plt.close()
        return tmp.name, float(vmin), float(vmax)

# Tabs
map_tab, finance_tab, score_tab = st.tabs(["🌍 Wind Map", "📊 Financial Dashboard", "📍 Site Score Map"])

# ------------------ Wind Map ------------------
with map_tab:
    st.title("Select a Region to Analyze Wind Potential")
    img_path, vmin, vmax = create_overlay(wind_data)
    map_obj = folium.Map(location=[germany.geometry.centroid.y.mean(), germany.geometry.centroid.x.mean()],
                         zoom_start=6, tiles="cartodbpositron")

    ImageOverlay(name="Wind Speed Heatmap", image=img_path,
                 bounds=[[transform[5] + transform[4]*wind_data.shape[0], transform[2]],
                         [transform[5], transform[2] + transform[0]*wind_data.shape[1]]],
                 opacity=0.6).add_to(map_obj)

    colormap = linear.viridis.scale(vmin, vmax)
    colormap.caption = 'Wind Speed (m/s)'
    colormap.add_to(map_obj)

    for i in range(0, wind_data.shape[0], 150):  # optimized sampling
        for j in range(0, wind_data.shape[1], 150):
            val = wind_data[i, j]
            if val <= 0:
                continue
            lat = transform[5] + i * transform[4]
            lon = transform[2] + j * transform[0]
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=None,
                fill=True,
                fill_opacity=0,
                tooltip=f"Wind Speed (m/s): {val:.2f}"
            ).add_to(map_obj)

    Draw(export=False, draw_options={"rectangle": {"shapeOptions": {"color": "blue"}}}).add_to(map_obj)
    st.session_state.st_data = st_folium(map_obj, use_container_width=True, height=500, returned_objects=["last_active_drawing"])

# ------------------ Financial Tab ------------------
with finance_tab:
    if "st_data" in st.session_state and st.session_state["st_data"].get("last_active_drawing"):
        polygon = st.session_state["st_data"]["last_active_drawing"]["geometry"]

        def extract_mean_in_polygon(polygon_geojson, raster):
            geom = shape(polygon_geojson)
            geom_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(raster.crs)
            out_image, _ = mask(raster, [mapping(geom_gdf.geometry.iloc[0])], crop=True)
            data = out_image[0]
            return np.nanmean(data[data > 0])

        with rasterio.open(wind_path) as wind_src:
            avg_wind_speed = extract_mean_in_polygon(polygon, wind_src)
        with rasterio.open(air_path) as air_src:
            avg_air_density = extract_mean_in_polygon(polygon, air_src)

        st.metric("Avg Wind Speed (m/s)", f"{avg_wind_speed:.2f}")
        st.metric("Avg Air Density (kg/m³)", f"{avg_air_density:.2f}")

        def financial_schedule(capacity_factor, air_density):
            lifetime = 20
            degradation = 0.5
            base_energy = 8760 * capacity_factor * installed_mw
            adjusted_energy = base_energy * (air_density / st.session_state.air_density_norm)
            annual_energy = [adjusted_energy * ((1 - degradation / 100) ** (y - 1)) for y in range(1, lifetime + 1)]
            revenue = [e * electricity_price for e in annual_energy]
            opex = [om_cost * installed_mw * ((1 + inflation_rate / 100) ** (y - 1)) for y in range(1, lifetime + 1)]
            net_cashflow = [r - o for r, o in zip(revenue, opex)]
            discounted = [n / ((1 + wacc / 100) ** y) for y, n in enumerate(net_cashflow, start=1)]
            return pd.DataFrame({
                "Year": list(range(1, lifetime + 1)),
                "Energy (MWh)": annual_energy,
                "Revenue (€)": revenue,
                "O&M (€)": opex,
                "Net Cash Flow (€)": net_cashflow,
                "Discounted Cash Flow (€)": discounted,
            })

        capacity_factor = min(max((avg_wind_speed - 3) / 9, 0), 1)
        df = financial_schedule(capacity_factor, avg_air_density)

        st.markdown(f"### 💰 NPV: €{df['Discounted Cash Flow (€)'].sum():,.0f}")
        st.markdown(f"### 📉 WACC: {wacc:.2f}%")

        st.altair_chart(alt.Chart(pd.melt(df, id_vars=["Year"], value_vars=["Net Cash Flow (€)", "Discounted Cash Flow (€)"]))
                        .mark_bar().encode(x="Year:O", y="value:Q", color="variable:N"), use_container_width=True)

        st.dataframe(df)
        st.download_button("📄 Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="financials.csv", mime="text/csv")

        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine="xlsxwriter")
        st.download_button("📊 Download Excel", data=excel_buffer.getvalue(), file_name="financials.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------ Site Score Tab ------------------
with score_tab:
    st.title("📍 Site Suitability Based on Wind Speed × Air Density")
    site_score_data = wind_data * air_data

    # Load and buffer restricted areas using projected CRS
    restricted_gdf = gpd.read_file(restricted_path).to_crs(crs)

    # Project to metric CRS for accurate buffering
    projected_crs = restricted_gdf.estimate_utm_crs()
    restricted_proj = restricted_gdf.to_crs(projected_crs)

    # Buffer around city centroids (e.g. 5 km)
    restricted_proj["geometry"] = restricted_proj.centroid.buffer(4000)

    # Reproject buffered geometries back to raster CRS
    restricted_buffered = restricted_proj.to_crs(crs)

    # Rasterize mask and apply it
    shapes = [(geom, 1) for geom in restricted_buffered.geometry]
    mask_array = rasterize(shapes, out_shape=site_score_data.shape, transform=transform, fill=0, dtype="uint8")
    site_score_data[mask_array == 1] = 0

    score_img_path, score_vmin, score_vmax = create_overlay(site_score_data)
    site_map = folium.Map(location=[germany.geometry.centroid.y.mean(), germany.geometry.centroid.x.mean()], zoom_start=6, tiles="cartodbpositron")

    ImageOverlay(name="Site Score", image=score_img_path,
                 bounds=[[transform[5] + transform[4]*site_score_data.shape[0], transform[2]],
                         [transform[5], transform[2] + transform[0]*site_score_data.shape[1]]],
                 opacity=0.6).add_to(site_map)

    colormap_score = linear.viridis.scale(score_vmin, score_vmax)
    colormap_score.caption = 'Site Score (Wind × Air Density, masked)'
    colormap_score.add_to(site_map)

    for i in range(0, site_score_data.shape[0], 150):
        for j in range(0, site_score_data.shape[1], 150):
            score = site_score_data[i, j]
            if score <= 0:
                continue
            lat = transform[5] + i * transform[4]
            lon = transform[2] + j * transform[0]
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=None,
                fill=True,
                fill_opacity=0,
                tooltip=f"Site Score: {score:.2f}"
            ).add_to(site_map)

    Draw(export=False, draw_options={"rectangle": {"shapeOptions": {"color": "red"}}}).add_to(site_map)
    st_folium(site_map, use_container_width=True, height=500)