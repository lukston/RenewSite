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
import pandas as pd
import altair as alt
from folium.raster_layers import ImageOverlay
from tempfile import NamedTemporaryFile
from folium.plugins import Draw
from rasterio.features import rasterize
from io import BytesIO
import requests

st.set_page_config(layout="wide")
st.title("Wind-Project Dashboard")


# ------------------ Load & Clean Energy-Price Scenarios from Google Sheets (XLSX export) ------------------

@st.cache_data(show_spinner="Downloading & parsing price scenarios…")
def load_price_scenarios_xlsx(sheet_id: str) -> pd.DataFrame:
    """
    1) Downloads the Google Sheet as XLSX via export?format=xlsx.
    2) Reads the FIRST worksheet into a DataFrame (dtype=str).
    3) Renames the first column to 'Scenario', strips whitespace from all
       column names and index names.
    4) Filters to only the three valid scenario rows: 'Upper Bound Scenario',
       'Base Case Scenario', 'Lower Bound Scenario'.
    5) Converts each year-column (e.g. '2024','2025',…) from string to float
       by replacing commas with dots.
    The sheet must be shared as “Anyone with the link can view.”
    """
    xlsx_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    try:
        resp = requests.get(xlsx_url)
        resp.raise_for_status()
    except Exception as e:
        st.error(
            "⚠️ Could not download the Google Sheet as XLSX. Please ensure:\n"
            "  1) The sheet is shared as “Anyone with the link can view.”\n"
            "  2) The sheet_id is correct.\n\n"
            f"Detailed error:\n{e}"
        )
        st.stop()

    xlsx_bytes = BytesIO(resp.content)
    try:
        df = pd.read_excel(xlsx_bytes, sheet_name=0, dtype=str)
    except Exception as e:
        st.error(
            "⚠️ Could not parse the downloaded XLSX. Verify that the first tab\n"
            "    indeed contains your three scenario rows and year columns.\n"
            f"Detailed error:\n{e}"
        )
        st.stop()

    # Rename first column to "Scenario" and strip whitespace in column names
    original_first_col = df.columns[0]
    df = df.rename(columns={original_first_col: "Scenario"})
    df.columns = df.columns.astype(str).str.strip()
    df["Scenario"] = df["Scenario"].astype(str).str.strip()
    df = df.set_index("Scenario", drop=True)
    df.index = df.index.str.strip()

    # Filter to exactly these three valid scenarios
    valid_scenarios = ["Upper Bound Scenario", "Base Case Scenario", "Lower Bound Scenario"]
    df = df.loc[df.index.intersection(valid_scenarios)]

    # Convert each year column to float (comma→dot, then to_numeric)
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .apply(lambda x: pd.to_numeric(x, errors="coerce"))
            .fillna(0.0)
        )

    return df


# Your Google Sheet ID (must be shared as “Anyone with the link can view”)
SHEET_ID = "1gbvX9IL_vkDDzl9feJvDzN-4oFMKWyNM"
price_scenarios_df = load_price_scenarios_xlsx(SHEET_ID)


# ------------------ Google Drive Download (Raster + GeoJSON) ------------------

@st.cache_data(show_spinner="Downloading raster/geojson from Drive…")
def gdrive_download(file_id: str, suffix: str) -> str:
    """
    Downloads a publicly‐shared Google Drive file by its file ID and returns
    a local temporary filename. The file must be shared as “Anyone with
    the link can view.”
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = requests.get(url)
    if not resp.ok or "text/html" in resp.headers.get("Content-Type", ""):
        st.error(
            "⚠️ Failed to download a valid file from Google Drive. "
            "Ensure that file is shared as “Anyone with link can view.”"
        )
        st.stop()
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    return tmp.name

# Replace these IDs if your Drive files change; they must be “Anyone with link can view”
WIND_FILE_ID       = "1cEMvR4O4Z2m6TyLXGxccoho0T7SL9_Pg"
AIR_FILE_ID        = "1IEympzffE3LZMMl1Hqs-HqGm2cQ9JokA"
RESTRICTED_FILE_ID = "1No-TTcys7wJl_GqPtTnEGTxvxVg_Tlb-"

wind_path       = gdrive_download(WIND_FILE_ID, ".tif")
air_path        = gdrive_download(AIR_FILE_ID, ".tif")
restricted_path = gdrive_download(RESTRICTED_FILE_ID, ".geojson")


# ------------------ Sidebar Inputs ------------------

st.sidebar.header("📈 Financial & Data Inputs")

unlevered_beta      = st.sidebar.number_input("Unlevered Beta",        0.0, 5.0,   0.8)
risk_free_rate      = st.sidebar.number_input("Risk-Free Rate (%)",     0.0, 10.0,  2.0)
market_risk_premium = st.sidebar.number_input("Market Risk Premium (%)",0.0, 15.0,  5.0)
cost_of_debt        = st.sidebar.slider("Cost of Debt (%)",       0.0, 15.0,  3.0, 0.1)
loan_pct            = st.sidebar.slider("Debt Ratio (%)",         0,   100,   70,  1)
tax_rate            = st.sidebar.slider("Corporate Tax Rate (%)", 0.0, 50.0,  25.0, 0.5)
inflation_rate      = st.sidebar.slider("Inflation Rate (%)",     0.0, 10.0,   2.0, 0.1)

# Add CAPEX input (in millions of €)
capex_million = st.sidebar.number_input("CAPEX (million €)", 0.0, 1000.0, 3.0, 0.1)
capex_eur     = capex_million * 1e6  # convert to €

# Only the three filtered scenarios appear
scenario = st.sidebar.selectbox(
    "Energy Price Scenario",
    options=price_scenarios_df.index.to_list(),
    index=price_scenarios_df.index.to_list().index("Base Case Scenario")
)

installed_mw = st.sidebar.slider("Installed Capacity (MW)", 1, 50, 5)
om_cost      = st.sidebar.number_input("O&M Cost (€/MW/year)", 0.0, 1e6, 40000.0)


# ------------------ WACC Calculation ------------------

levered_beta           = unlevered_beta * (1 + loan_pct / 100)
debt_ratio             = min(max(loan_pct / 100, 0), 1)
equity_ratio           = 1 - debt_ratio
cost_of_equity         = risk_free_rate + levered_beta * market_risk_premium
after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate / 100)
wacc = equity_ratio * cost_of_equity + debt_ratio * after_tax_cost_of_debt
wacc = max(wacc, 0)  # ensure non-negative


# ------------------ Helper to calculate IRR manually using numpy.roots ------------------

def calculate_irr(cashflows: np.ndarray) -> float:
    """
    Calculates IRR by finding the real positive root of the polynomial:
      CF_0 * x^(n-1) + CF_1 * x^(n-2) + ... + CF_(n-1) * x^0 = 0,
    where x = 1 + IRR.
    Returns IRR as a decimal (e.g. 0.085 for 8.5%). If no valid root, returns np.nan.
    """
    coeffs = cashflows.copy().astype(float)
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    if not real_roots:
        return np.nan
    best_root = min(real_roots, key=lambda x: abs(x - 1.0))
    return best_root - 1.0


# ------------------ Load rasters and Germany outline ------------------

def load_rasters(wind_file: str, air_file: str):
    germany = gpd.read_file(
        "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/DEU.geo.json"
    )
    with rasterio.open(wind_file) as wind_src, rasterio.open(air_file) as air_src:
        germany = germany.to_crs(wind_src.crs)
        return (
            wind_src.read(1),
            air_src.read(1),
            wind_src.transform,
            germany,
            wind_src.crs
        )

wind_data, air_data, transform, germany, crs = load_rasters(wind_path, air_path)
wind_data = np.nan_to_num(wind_data, nan=0.0, posinf=0.0, neginf=0.0)
air_data  = np.nan_to_num(air_data,  nan=0.0, posinf=0.0, neginf=0.0)


# ------------------ Helper: Create a transparent PNG overlay from a raster ------------------

def create_overlay(data: np.ndarray):
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    norm        = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    cmap        = plt.get_cmap("viridis")
    rgba        = cmap(norm)
    rgba[..., 3] = (data > 0).astype(float)

    fig, ax = plt.subplots()
    ax.imshow(rgba, origin="upper")
    ax.axis("off")

    with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches="tight", pad_inches=0, dpi=150, transparent=True)
        plt.close()
        return tmp.name, float(vmin), float(vmax)


# ------------------ Create Tabs ------------------

map_tab, finance_tab, price_tab, score_tab, table_tab = st.tabs([
    "🌍 Wind Map",
    "📊 Financial Dashboard",
    "⚡ Energy Prices",
    "📍 Site Score Map",
    "📋 Site Score Table"
])


# ------------------ 1) Wind Map Tab ------------------
with map_tab:
    st.markdown("## Draw a polygon to pick a site")
    img_path, vmin, vmax = create_overlay(wind_data)

    folium_map = folium.Map(
        location=[germany.geometry.centroid.y.mean(), germany.geometry.centroid.x.mean()],
        zoom_start=6,
        tiles="cartodbpositron",
        width="100%"
    )

    ImageOverlay(
        name="Wind Speed Heatmap",
        image=img_path,
        bounds=[
            [transform[5] + transform[4] * wind_data.shape[0], transform[2]],
            [transform[5], transform[2] + transform[0] * wind_data.shape[1]]
        ],
        opacity=0.6
    ).add_to(folium_map)

    colormap = linear.viridis.scale(vmin, vmax)
    colormap.caption = "Wind Speed (m/s)"
    colormap.add_to(folium_map)

    for i in range(0, wind_data.shape[0], 150):
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
                tooltip=f"{val:.2f} m/s"
            ).add_to(folium_map)

    Draw(
        export=False,
        draw_options={"rectangle": {"shapeOptions": {"color": "blue"}}}
    ).add_to(folium_map)

    st.session_state.st_data = st_folium(
        folium_map,
        use_container_width=True,
        height=700,
        returned_objects=["last_active_drawing"]
    )


# ------------------ 2) Financial Dashboard Tab ------------------
with finance_tab:
    if "st_data" in st.session_state and st.session_state["st_data"].get("last_active_drawing"):
        polygon = st.session_state["st_data"]["last_active_drawing"]["geometry"]

        def extract_mean_in_polygon(geojson_poly, raster):
            geom = shape(geojson_poly)
            geom_gdf = (
                gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
                .to_crs(raster.crs)
            )
            out_image, _ = mask(
                raster,
                [mapping(geom_gdf.geometry.iloc[0])],
                crop=True
            )
            data = out_image[0]
            return np.nanmean(data[data > 0])

        with rasterio.open(wind_path) as ws:
            avg_wind_speed = extract_mean_in_polygon(polygon, ws)
        with rasterio.open(air_path) as ad:
            avg_air_density = extract_mean_in_polygon(polygon, ad)

        st.metric("Avg Wind Speed (m/s)", f"{avg_wind_speed:.2f}")
        st.metric("Avg Air Density (kg/m³)", f"{avg_air_density:.2f}")

        # Display the formula in Markdown (using a raw‐string so backslashes are preserved)
        st.markdown(
            r"""
            ### ⚙️ Power‐Estimate Formula

            Based on the three most popular wind turbines, this formula estimates the expected annual energy (MWh):

            $$
            P_{\mathrm{est}} = 0.5\,\rho\,A_{\mathrm{avg}}\,V^3\,C_{p,\mathrm{avg}}
            \approx 3{,}848.37\,\rho\,V^3
            \quad(\text{Watts})
            $$

            Equivalently, in megawatts:

            $$
            P_{\mathrm{est}}\,[\mathrm{MW}]
            = 3.84837 \times 10^{-3}\,\rho\,V^3
            $$

            Where:
            - $\rho$ = Average Air Density (kg/m³)  
            - $A_{\mathrm{avg}}$ = Average rotor swept area (m²)  
            - $V$ = Average wind speed (m/s)  
            - $C_{p,\mathrm{avg}}$ = Average power coefficient (≈ 0.45)

            Finally, multiplying $P_{\mathrm{est}}\,[\mathrm{MW}]$ by 8 760 hours/year yields the annual energy in *MWh/year*.
            """
        )

        # Compute P_est (MW) using the formula and then MWh/year
        P_est_MW = 3.84837e-3 * avg_air_density * (avg_wind_speed ** 3)
        annual_energy_MWh = P_est_MW * 8760  # hours in a year

        st.metric("Pₑₛₜ (MW)", f"{P_est_MW:.2f}")
        st.metric("Annual Energy (MWh)", f"{annual_energy_MWh:,.0f}")

        # Build a 20‐period price series so that:
        #   Period 0 → CAPEX only (no revenue),
        #   Period 1 → price for 2025,
        #   Period 2 → price for 2026,
        #   …,
        #   Period 19 → price for 2043.
        lifetime = 20
        price_series = [0.0] * lifetime
        for p in range(1, lifetime):
            year_label = str(2024 + p)  # p=1→"2025", p=2→"2026", …, p=19→"2043"
            if year_label in price_scenarios_df.columns:
                price_series[p] = price_scenarios_df.at[scenario, year_label]
            else:
                price_series[p] = 0.0

        def financial_schedule(P_est_MWh_per_year, prices, capex):
            """
            20‐period schedule:
            - Period 0: CAPEX outflow only.
            - Period 1: revenue = P_est_MWh_per_year * price[1], minus O&M.
            - Period p: revenue = P_est_MWh_per_year * price[p], minus O&M.
            """
            cf = [0.0] * lifetime
            # Period 0 = CAPEX
            cf[0] = -capex

            # Periods 1..19: revenue and O&M
            for period in range(1, lifetime):
                revenue = P_est_MWh_per_year * prices[period]
                opex = om_cost * installed_mw * ((1 + inflation_rate / 100) ** (period - 1))
                cf[period] = revenue - opex

            # Discounted CF and cumulative
            dcf = [cf[p] / ((1 + wacc / 100) ** p) for p in range(lifetime)]
            cum_dcf = np.cumsum(dcf)

            return pd.DataFrame({
                "Year":         list(range(0, lifetime)),    # 0..19
                "Price (€)":    prices,                      # P0=0, P1=2025, P2=2026, …
                "Energy (MWh)": [0.0] + [P_est_MWh_per_year] * (lifetime - 1),
                "Revenue (€)":  [0.0] + [P_est_MWh_per_year * prices[p] for p in range(1, lifetime)],
                "O&M (€)":      [0.0] + [om_cost * installed_mw * ((1 + inflation_rate / 100) ** (p - 1))
                                         for p in range(1, lifetime)],
                "Net Cash Flow (€)":          cf,
                "Discounted Cash Flow (€)":   dcf,
                "Cumulative DCF (€)":         cum_dcf,
            })

        df = financial_schedule(annual_energy_MWh, price_series, capex_eur)

        # Base‐Case NPV
        npv = df["Discounted Cash Flow (€)"].sum()

        # IRR = solves sum(CF_t / (1+IRR)^t) = 0
        irr = calculate_irr(df["Net Cash Flow (€)"].values)

        st.markdown(f"### 💰 NPV: €{npv:,.0f}")
        if not np.isnan(irr):
            st.markdown(f"### 🔁 IRR: {irr * 100:.2f}%")
        else:
            st.markdown("### 🔁 IRR: N/A")
        st.markdown(f"### ⚖️ WACC: {wacc:.2f}%")

        # — Discounted CF Bars + Cumulative DCF Line —
        bars = (
            alt.Chart(df)
            .mark_bar(color="#1f77b4")
            .encode(
                x=alt.X("Year:O", title="Year (0=CAPEX, 1=2025, 2=2026, …)"),
                y=alt.Y("Discounted Cash Flow (€):Q", title="Discounted Cash Flow (€)")
            )
        )
        line = (
            alt.Chart(df)
            .mark_line(color="red", point=True)
            .encode(
                x="Year:O",
                y=alt.Y("Cumulative DCF (€):Q", title="Cumulative Discounted CF (€)")
            )
        )
        layered_chart = alt.layer(bars, line).resolve_scale(y="independent").properties(
            height=400,
            title="Discounted CF (blue bars) & Cumulative DCF (red line)"
        )
        st.altair_chart(layered_chart, use_container_width=True)

        # — NPV vs WACC Sensitivity —
        net_cf_series = df["Net Cash Flow (€)"].values
        wacc_vals = np.arange(0.0, 0.201, 0.005)
        npv_sens = []
        for w in wacc_vals:
            discount_factors = np.array([(1 + w) ** p for p in range(len(net_cf_series))])
            npv_val = np.sum(net_cf_series / discount_factors)
            npv_sens.append(npv_val)
        sens_df = pd.DataFrame({
            "WACC (%)": (wacc_vals * 100),
            "NPV (€)":  npv_sens
        })
        sens_chart = (
            alt.Chart(sens_df)
            .mark_line(point=True, color="#2ca02c")
            .encode(
                x=alt.X("WACC (%):Q", title="WACC (%)"),
                y=alt.Y("NPV (€):Q",   title="NPV (€)")
            )
            .properties(title="NPV Sensitivity to WACC (0%–20%)")
        )
        st.altair_chart(sens_chart, use_container_width=True)

        # Show the full DataFrame and downloads
        st.dataframe(df)
        st.download_button(
            "📄 Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="financials.csv",
            mime="text/csv"
        )
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine="xlsxwriter")
        st.download_button(
            "📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name="financials.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ------------------ 3) Energy Prices Tab ------------------
with price_tab:
    st.markdown("## Energy-Price Forecasts (2024–2043) by Scenario")
    lifetime = 20
    start_year = 2024
    scenario_cols = [str(start_year + p) for p in range(lifetime)]
    price_chart_df = (
        price_scenarios_df[scenario_cols]
        .reset_index()
        .melt(
            id_vars="Scenario",
            var_name="YearLabel",
            value_name="Price"
        )
    )
    price_chart_df["YearNum"] = price_chart_df["YearLabel"].astype(int) - start_year

    price_line = (
        alt.Chart(price_chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("YearNum:O", title="Year (0=2024, 1=2025, …)"),
            y=alt.Y("Price:Q", title="Price (€)"),
            color=alt.Color("Scenario:N", title="Scenario")
        )
        .properties(title="Energy-Price Forecasts (2024–2043) by Scenario")
    )
    st.altair_chart(price_line, use_container_width=True)


# ------------------ 4) Site Score Tab ------------------
with score_tab:
    st.markdown("## Site Suitability (Wind Speed × Air Density)")
    site_score_data = wind_data * air_data

    restricted_gdf = gpd.read_file(restricted_path).to_crs(crs)
    projected_crs = restricted_gdf.estimate_utm_crs()
    restricted_proj = restricted_gdf.to_crs(projected_crs)

    restricted_proj["geometry"] = restricted_proj.centroid.buffer(4000)  # 4 km buffer
    restricted_buffered = restricted_proj.to_crs(crs)

    shapes = [(geom, 1) for geom in restricted_buffered.geometry]
    mask_array = rasterize(
        shapes,
        out_shape=site_score_data.shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )
    site_score_data[mask_array == 1] = 0

    score_img_path, score_vmin, score_vmax = create_overlay(site_score_data)
    site_map = folium.Map(
        location=[germany.geometry.centroid.y.mean(), germany.geometry.centroid.x.mean()],
        zoom_start=6,
        tiles="cartodbpositron",
        width="100%"
    )

    ImageOverlay(
        name="Site Score",
        image=score_img_path,
        bounds=[
            [transform[5] + transform[4] * site_score_data.shape[0], transform[2]],
            [transform[5], transform[2] + transform[0] * site_score_data.shape[1]]
        ],
        opacity=0.6
    ).add_to(site_map)

    colormap_score = linear.viridis.scale(score_vmin, score_vmax)
    colormap_score.caption = "Site Score (Wind × Air Density, masked)"
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
                tooltip=f"{score:.2f}"
            ).add_to(site_map)

    Draw(
        export=False,
        draw_options={"rectangle": {"shapeOptions": {"color": "red"}}}
    ).add_to(site_map)

    st_folium(site_map, use_container_width=True, height=700)


# ------------------ 5) Site Score Table Tab ------------------
with table_tab:
    st.markdown("## Site Score Table (Sorted by Score)")

    coords = []
    for i in range(site_score_data.shape[0]):
        for j in range(site_score_data.shape[1]):
            lat = transform[5] + i * transform[4]
            lon = transform[2] + j * transform[0]
            score = site_score_data[i, j]
            if mask_array[i, j] == 1:
                coords.append((lat, lon, "restricted"))
            elif score > 0:
                coords.append((lat, lon, round(score, 2)))

    site_df = pd.DataFrame(coords, columns=["Latitude", "Longitude", "Site Score"])
    site_df_sorted = site_df.copy()
    site_df_sorted["SortScore"] = pd.to_numeric(site_df_sorted["Site Score"], errors="coerce")
    site_df_sorted = site_df_sorted.sort_values("SortScore", ascending=False).drop(columns="SortScore").reset_index(drop=True)

    top_n = st.number_input("Show top N sites by score", min_value=10, max_value=10000, value=500)
    st.dataframe(site_df_sorted.head(top_n))

    st.download_button(
        "📥 Download Site Scores (CSV)",
        data=site_df_sorted.to_csv(index=False).encode("utf-8"),
        file_name="site_scores_sorted.csv",
        mime="text/csv"
    )