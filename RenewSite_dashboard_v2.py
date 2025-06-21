from shapely.geometry import mapping, shape
import streamlit as st
import folium
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from branca.colormap import linear
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
from folium.raster_layers import ImageOverlay
from tempfile import NamedTemporaryFile
from folium.plugins import Draw
from io import BytesIO
import requests

import plotly.graph_objects as go
import feedparser  # pip install feedparser

st.set_page_config(layout="wide")
# Updated logo with Google Drive thumbnail endpoint
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:1em; margin-bottom:1em;">
        <img src="https://drive.google.com/thumbnail?id=1MjFnHO_Z2SWS3C6sLuBu69MM_CaIvyBA" alt="RenewSite Logo" style="height:60px;">
        <h1 style="margin:0;">Wind-Project Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# st.title("Wind-Project Dashboard")

# ------------------ Load & Clean Energy-Price Scenarios ------------------

@st.cache_data(show_spinner="Downloading & parsing price scenarios…")
def load_price_scenarios_xlsx(sheet_id: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    resp = requests.get(url); resp.raise_for_status()
    df = pd.read_excel(BytesIO(resp.content), sheet_name=0, dtype=str)
    first = df.columns[0]
    df = df.rename(columns={first: "Scenario"})
    df.columns = df.columns.map(lambda c: str(c).strip())
    df["Scenario"] = df["Scenario"].astype(str).str.strip()
    df = df.set_index("Scenario", drop=True)
    valid = ["Upper Bound Scenario", "Base Case Scenario", "Lower Bound Scenario"]
    df = df.loc[df.index.intersection(valid)]
    for col in df.columns:
        df[col] = (
            df[col]
            .apply(lambda x: str(x).replace(",", ".").strip())
            .pipe(lambda s: pd.to_numeric(s, errors="coerce"))
            .fillna(0.0)
        )
    return df

SHEET_ID = "1gbvX9IL_vkDDzl9feJvDzN-4oFMKWyNM"
price_scenarios_df = load_price_scenarios_xlsx(SHEET_ID)

# ------------------ Google Drive Download ------------------

@st.cache_data(show_spinner="Downloading raster/geojson from Drive…")
def gdrive_download(file_id: str, suffix: str) -> str:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = requests.get(url); resp.raise_for_status()
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    return tmp.name

WIND_FILE_ID = "1cEMvR4O4Z2m6TyLXGxccoho0T7SL9_Pg"
AIR_FILE_ID  = "1IEympzffE3LZMMl1Hqs-HqGm2cQ9JokA"
wind_path    = gdrive_download(WIND_FILE_ID, ".tif")
air_path     = gdrive_download(AIR_FILE_ID,  ".tif")

# ------------------ Turbine Catalog ------------------

TURBINE_CATALOG = [
    {"name":"Enercon E-82","price_eur":2_600_000,"rotor_diameter_m":82,"rated_power_kw":2300,"cp":0.45},
    {"name":"Siemens SG 4.5-132","price_eur":3_300_000,"rotor_diameter_m":132,"rated_power_kw":4500,"cp":0.47},
    {"name":"Vestas V150-6.0","price_eur":4_000_000,"rotor_diameter_m":150,"rated_power_kw":6000,"cp":0.48},
]

# ------------------ Sidebar Inputs ------------------

st.sidebar.header("📈 Financial & Data Inputs")
unlevered_beta      = st.sidebar.number_input("Unlevered Beta",         0.0, 5.0,   0.8)
risk_free_rate      = st.sidebar.number_input("Risk-Free Rate (%)",     0.0,10.0,   2.0)
market_risk_premium = st.sidebar.number_input("Market Risk Premium (%)",0.0,15.0,   5.0)
cost_of_debt        = st.sidebar.slider("Cost of Debt (%)",       0.0,15.0,   3.0,0.1)
loan_pct            = st.sidebar.slider("Debt Ratio (%)",         0,  100,    70,1)
tax_rate            = st.sidebar.slider("Corporate Tax Rate (%)", 0.0,50.0,  25.0,0.5)
inflation_rate      = st.sidebar.slider("Inflation Rate (%)",     0.0,10.0,   2.0,0.1)

turbine_names = [t["name"] for t in TURBINE_CATALOG]
selected      = st.sidebar.selectbox("Turbine Model", turbine_names)
turbine       = next(t for t in TURBINE_CATALOG if t["name"]==selected)
n_turbines    = st.sidebar.number_input("Number of Turbines", 1, 100, 3, 1)
capex_eur     = turbine["price_eur"] * n_turbines
st.sidebar.markdown(f"**CAPEX (auto):** €{capex_eur:,.0f}")

scenario = st.sidebar.selectbox(
    "Energy Price Scenario",
    options=price_scenarios_df.index.tolist(),
    index=price_scenarios_df.index.tolist().index("Base Case Scenario")
)
om_cost = st.sidebar.number_input("O&M Cost (€/MW/year)",0.0,1e6,40000.0)

# ------------------ Profitability Metrics Helpers ------------------

levered_beta        = unlevered_beta * (1 + loan_pct/100)
debt_ratio          = loan_pct/100
equity_ratio        = 1 - debt_ratio
cost_of_equity      = risk_free_rate + levered_beta * market_risk_premium
after_tax_debt_cost = cost_of_debt * (1 - tax_rate/100)
wacc                = equity_ratio*cost_of_equity + debt_ratio*after_tax_debt_cost
wacc = max(wacc,0.0)

# --- MIRR, Payback, PI ---
def calculate_mirr(cashflows, finance_rate, reinvest_rate):
    cashflows = np.array(cashflows, dtype=float)
    n = len(cashflows)
    positive = np.where(cashflows > 0, cashflows, 0)
    negative = np.where(cashflows < 0, cashflows, 0)
    pv_neg = np.sum(negative / ((1 + finance_rate) ** np.arange(n)))
    fv_pos = np.sum(positive * ((1 + reinvest_rate) ** (n - 1 - np.arange(n))))
    if pv_neg == 0 or fv_pos == 0:
        return np.nan
    mirr = (abs(fv_pos / pv_neg)) ** (1/(n-1)) - 1
    return mirr

def calculate_payback_period(cashflows):
    cumsum = np.cumsum(cashflows)
    payback_years = np.argmax(cumsum >= 0)
    if cumsum[payback_years] < 0:
        return np.nan  # Never paid back
    return payback_years

# ------------------ Load Rasters & Germany Outline ------------------

def load_rasters(wf, af):
    germany = gpd.read_file(
        "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/DEU.geo.json"
    ).to_crs("EPSG:25832")
    with rasterio.open(wf) as ws, rasterio.open(af) as asrc:
        germany = germany.to_crs(ws.crs)
        return ws.read(1), asrc.read(1), ws.transform, germany, ws.crs

wind_data, air_data, transform, germany, crs = load_rasters(wind_path, air_path)
wind_data = np.nan_to_num(wind_data, nan=0.0, posinf=0.0, neginf=0.0)
air_data  = np.nan_to_num(air_data,  nan=0.0, posinf=0.0, neginf=0.0)

# ------------------ Overlay Helper ------------------

def create_overlay(data: np.ndarray):
    vmin, vmax = float(data.min()), float(data.max())
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap("viridis")
    rgba = cmap(norm)
    rgba[..., 3] = (data > 0).astype(float)
    fig, ax = plt.subplots()
    ax.imshow(rgba, origin="upper")
    ax.axis("off")
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches="tight", pad_inches=0,
                    dpi=150, transparent=True)
    plt.close(fig)
    return tmp.name, vmin, vmax

# ------------------ Tabs ------------------

map_tab, finance_tab, site_tab, news_tab, lcoe_tab, table_tab = st.tabs([
    "🌍 Wind Map", "📊 Financial Dashboard",
    "📍 Site Dashboard",
    "📰 News", "🧮 LCOE Calculator",
    "📄 Site Score Table"
])

# --- 1) Wind Map ---
with map_tab:
    st.markdown("## Draw a polygon to pick a site")

    # ---------------- Naturschutzgebiete ----------------
    NSG_FILE_ID = "1p154rcpW4MDNcfov-LIJ33SQjKv7mfIK"
    nsg_path = gdrive_download(NSG_FILE_ID, ".geojson")
    nsg = gpd.read_file(nsg_path)
    
    for col in nsg.columns:
        if pd.api.types.is_datetime64_any_dtype(nsg[col]):
            nsg[col] = nsg[col].astype(str)
    if nsg.crs is not None and nsg.crs.to_epsg() != 4326:
        nsg = nsg.to_crs(epsg=4326)

    # ---------------- Score Map (Site Suitability) ----------------
    SCORE_FILE_ID = "17EZxRok4zRmS7iX1Y1aR7Znh2RpnvLmd"
    score_path    = gdrive_download(SCORE_FILE_ID, ".tif")
    with rasterio.open(score_path) as score_src:
        sd = np.nan_to_num(score_src.read(1), nan=0., posinf=0., neginf=0.)
        score_transform = score_src.transform
        score_overlay, smin, smax = create_overlay(sd)
    from rasterio.warp import transform_bounds
    score_bounds = transform_bounds(
        crs, "EPSG:4326",
        score_transform[2],
        score_transform[5] + score_transform[4]*sd.shape[0],
        score_transform[2] + score_transform[0]*sd.shape[1],
        score_transform[5]
    )

    # ---------------- Wind Map Overlay ----------------
    img, vmin, vmax = create_overlay(wind_data)
    wind_bounds = transform_bounds(
        crs, "EPSG:4326",
        transform[2],
        transform[5] + transform[4] * wind_data.shape[0],
        transform[2] + transform[0] * wind_data.shape[1],
        transform[5]
    )

    # ---------------- Karte erstellen ----------------
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, tiles="cartodbpositron", width="100%")

    # Site Score Overlay
    ImageOverlay(
        image=score_overlay,
        bounds=[[score_bounds[1], score_bounds[0]], [score_bounds[3], score_bounds[2]]],
        opacity=0.4,
        name="Site Suitability"
    ).add_to(m)

    # Wind Overlay
    ImageOverlay(
        image=img,
        bounds=[[wind_bounds[1], wind_bounds[0]], [wind_bounds[3], wind_bounds[2]]],
        opacity=0.6,
        name="Wind Speed"
    ).add_to(m)

    # Wind Tooltips
    for i in range(0, wind_data.shape[0], 150):
        for j in range(0, wind_data.shape[1], 150):
            v = wind_data[i, j]
            if v > 0:
                lat = transform[5] + i * transform[4]
                lon = transform[2] + j * transform[0]
                folium.CircleMarker(
                    [lat, lon], radius=10, stroke=False,
                    fill=True, fill_color="transparent", fill_opacity=0,
                    tooltip=f"{v:.2f} m/s"
                ).add_to(m)

    # GeoJSON-Layer für Naturschutz
    folium.GeoJson(
        data=nsg.__geo_interface__,
        name="Naturschutzgebiete",
        style_function=lambda feature: {
            "color": "#FF0000",
            "fillColor": "#FF0000",
            "weight": 1,
            "fillOpacity": 0.3,
        }
    ).add_to(m)

    # Interaktive Auswahl
    Draw(export=False,
         draw_options={"rectangle": {"shapeOptions": {"color": "blue"}}}
    ).add_to(m)

    # Colorbars (Legenden)
    cb1 = linear.viridis.scale(smin, smax)
    cb1.caption = "Site Score"
    cb1.add_to(m)

    cb2 = linear.viridis.scale(vmin, vmax)
    cb2.caption = "Wind Speed"
    cb2.add_to(m)

    # Karte anzeigen
    folium.LayerControl().add_to(m)
    st.session_state.st_data = st_folium(
        m, height=700, use_container_width=True,
        returned_objects=["last_active_drawing"]
    )

 # --- 2) Financial Dashboard ---

# --- Site Dashboard ---
with site_tab:
    st.markdown("## 📍 Site Dashboard")

    if not st.session_state.get("st_data", {}).get("last_active_drawing"):
        st.info("Draw a site polygon on the Wind Map first.")
    else:
        poly = st.session_state["st_data"]["last_active_drawing"]["geometry"]

        def extract_mean(poly_geojson, rast):
            geom = shape(poly_geojson)
            gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(rast.crs)
            arr, _ = mask(rast, [mapping(gdf.geometry.iloc[0])], crop=True)
            vals = arr[0][arr[0] > 0]
            return float(np.nanmean(vals)) if vals.size else 0.0

        with rasterio.open(wind_path) as ws:
            avg_wind = extract_mean(poly, ws)
        with rasterio.open(air_path) as ad:
            avg_air = extract_mean(poly, ad)
        SCORE_FILE_ID = "17EZxRok4zRmS7iX1Y1aR7Znh2RpnvLmd"
        score_path = gdrive_download(SCORE_FILE_ID, ".tif")
        with rasterio.open(score_path) as ss:
            site_score = extract_mean(poly, ss)

        # --- Helper for interpolated color (red-yellow-green) ---
        def get_color(val, min_val, max_val):
            ratio = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
            ratio = max(0, min(1, ratio))
            if ratio < 0.5:
                r = 255
                g = int(255 * ratio * 2)
            else:
                r = int(255 * (1 - (ratio - 0.5) * 2))
                g = 255
            return f"rgb({r},{g},0)"

        col1, col2, col3 = st.columns(3)
        with col1:
            # Wind Speed Gauge
            min_wind, max_wind = 0, 13
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_wind,
                title={'text': "Wind Speed (m/s)"},
                gauge={
                    'axis': {'range': [min_wind, max_wind]},
                    'bar': {'color': get_color(avg_wind, min_wind, max_wind)},
                    'steps': []
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig1.update_traces(gauge={
                'axis': {'range': [min_wind, max_wind]},
                'bar': {'color': get_color(avg_wind, min_wind, max_wind), 'thickness': 0.2},
                'bgcolor': "rgba(0,0,0,0)",
                'threshold': {
                    'line': {'color': "black", 'width': 0},
                    'thickness': 0.0,
                    'value': avg_wind
                }
            })
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "black"},
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            # Air Density Gauge
            min_air, max_air = 0, 1.5
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_air,
                title={'text': "Air Density (kg/m³)"},
                gauge={
                    'axis': {'range': [min_air, max_air]},
                    'bar': {'color': get_color(avg_air, min_air, max_air)},
                    'steps': []
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig2.update_traces(gauge={
                'axis': {'range': [min_air, max_air]},
                'bar': {'color': get_color(avg_air, min_air, max_air), 'thickness': 0.2},
                'bgcolor': "rgba(0,0,0,0)",
                'threshold': {
                    'line': {'color': "black", 'width': 0},
                    'thickness': 0.0,
                    'value': avg_air
                }
            })
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "black"},
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig2, use_container_width=True)
        with col3:
            # Site Score Gauge
            min_score, max_score = 0, 15
            fig3 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=site_score,
                title={'text': "Site Score"},
                gauge={
                    'axis': {'range': [min_score, max_score]},
                    'bar': {'color': get_color(site_score, min_score, max_score)},
                    'steps': []
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig3.update_traces(gauge={
                'axis': {'range': [min_score, max_score]},
                'bar': {'color': get_color(site_score, min_score, max_score), 'thickness': 0.2},
                'bgcolor': "rgba(0,0,0,0)",
                'threshold': {
                    'line': {'color': "black", 'width': 0},
                    'thickness': 0.0,
                    'value': site_score
                }
            })
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "black"},
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig3, use_container_width=True)

        # --- CO₂ Avoided Content (moved from CO2 tab) ---
        st.markdown("## 🌱 Estimated CO₂ Emissions Avoided")
        def extract_mean(poly_geojson, rast):
            geom = shape(poly_geojson)
            gdf = gpd.GeoDataFrame(
                geometry=[geom], crs="EPSG:4326"
            ).to_crs(rast.crs)
            arr, _ = mask(rast, [mapping(gdf.geometry.iloc[0])], crop=True)
            vals = arr[0][arr[0] > 0]
            return float(np.nanmean(vals)) if vals.size else 0.0

        with rasterio.open(wind_path) as ws:
            avg_wind = extract_mean(poly, ws)
        with rasterio.open(air_path) as ad:
            avg_air = extract_mean(poly, ad)

        D, Cp = turbine["rotor_diameter_m"], turbine["cp"]
        P_W     = 0.5 * avg_air * np.pi*(D/2)**2 * (avg_wind**3) * Cp
        P_MW    = P_W / 1e6
        cap_mw  = turbine["rated_power_kw"] / 1000
        avg_P   = min(P_MW, cap_mw)
        total_MW, total_MWh = avg_P * n_turbines, avg_P * 8760 * n_turbines

        # *** Even more pessimistic availability ***
        availability     = 0.85  # "P90"/pessimistic
        ramp_up          = [0.75] + [1.0]*19
        degradation      = [(1 - 0.01*t) for t in range(20)]  # Faster degradation: 1%/yr
        lifetime = 20

        # Use Year 1 MWh for current offset (worst case)
        eff_MWh_yr1 = total_MWh * availability * ramp_up[1] * degradation[1]
        coal_factor = 0.75   # t CO2/MWh
        gas_factor  = 0.40   # t CO2/MWh
        coal_offset = eff_MWh_yr1 * coal_factor
        gas_offset  = eff_MWh_yr1 * gas_factor

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Annual CO₂ offset vs Coal**")
            st.markdown(f"<span style='font-size:2.4em; font-weight:bold'>{coal_offset:,.0f} t/y</span>", unsafe_allow_html=True)
        with col2:
            st.markdown("**Annual CO₂ offset vs Gas**")
            st.markdown(f"<span style='font-size:2.4em; font-weight:bold'>{gas_offset:,.0f} t/y</span>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🌍 What does that mean?")

        # --- Pessimistic equivalency factors (EU/IPCC/Conservative) ---
        cars = coal_offset / 2.5      # very small EU car, pessimistic
        flights = coal_offset / 0.9   # economy flight EU-NY, pessimistic
        trees = coal_offset / 0.008   # slow-growing tree

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div style='font-size:3em; text-align:center;'>🚗</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2em; font-weight:bold; text-align:center'>{int(cars):,}</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center;'>Cars off road (1yr)</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='font-size:3em; text-align:center;'>✈️</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2em; font-weight:bold; text-align:center'>{int(flights):,}</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center;'>Transatlantic flights avoided</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div style='font-size:3em; text-align:center;'>🌳</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2em; font-weight:bold; text-align:center'>{int(trees):,}</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center;'>Mature trees planted (1yr)</div>", unsafe_allow_html=True)
        st.caption(
            "Equivalency sources: [EPA](https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator), IPCC, EU averages. Factors are highly conservative."
        )
with finance_tab:
    if not st.session_state.get("st_data", {}).get("last_active_drawing"):
        st.info("Draw a site polygon on the Wind Map first.")
    else:
        poly = st.session_state["st_data"]["last_active_drawing"]["geometry"]

        def extract_mean(poly_geojson, rast):
            geom = shape(poly_geojson)
            gdf = gpd.GeoDataFrame(
                geometry=[geom], crs="EPSG:4326"
            ).to_crs(rast.crs)
            arr, _ = mask(rast, [mapping(gdf.geometry.iloc[0])], crop=True)
            vals = arr[0][arr[0] > 0]
            return float(np.nanmean(vals)) if vals.size else 0.0

        with rasterio.open(wind_path) as ws:
            avg_wind = extract_mean(poly, ws)
        with rasterio.open(air_path) as ad:
            avg_air = extract_mean(poly, ad)

        D, Cp = turbine["rotor_diameter_m"], turbine["cp"]
        P_W     = 0.5 * avg_air * np.pi*(D/2)**2 * (avg_wind**3) * Cp
        P_MW    = P_W / 1e6
        cap_mw  = turbine["rated_power_kw"] / 1000
        avg_P   = min(P_MW, cap_mw)
        total_MW, total_MWh = avg_P * n_turbines, avg_P * 8760 * n_turbines

        availability     = 0.95
        ramp_up          = [0.75] + [1.0]*19
        degradation      = [(1 - 0.005*t) for t in range(20)]
        debt_amt         = capex_eur * debt_ratio
        equity_amt       = capex_eur * (1 - debt_ratio)
        lifetime = 20
        prices   = [0.0]*lifetime
        for t in range(1, lifetime):
            yr = str(2024 + t)
            prices[t] = price_scenarios_df.at[scenario, yr] \
                        if yr in price_scenarios_df.columns else 0.0
        depreciation = capex_eur / lifetime

        cost_of_equity_rate = cost_of_equity / 100
        interest_exp = debt_amt * (cost_of_debt/100)
        # Prepare lists for all intermediate steps
        year_list = []
        eff_MWh_list = []
        revenue_list = []
        opex_list = []
        capex_list = []
        depreciation_list = []
        interest_list = []
        ebit_list = []
        ebt_list = []
        tax_list = []
        net_income_list = []
        debt_repay_list = []
        net_borrow_list = []
        fcfe_list = []
        disc_fcfe_list = []
        cumulative_fcfe_list = []

        for t in range(lifetime):
            if t == 0:
                eff_MWh = 0
                revenue = 0
                opex = 0
                capex = capex_eur
                depreciation_t = 0
                net_borrowing = debt_amt
                interest_t = 0
                debt_repayment = 0
            else:
                eff_MWh = total_MWh * availability * ramp_up[t] * degradation[t]
                revenue = eff_MWh * prices[t]
                opex = om_cost * total_MW * ((1 + inflation_rate/100)**(t-1))
                capex = 0
                depreciation_t = depreciation
                net_borrowing = 0
                interest_t = interest_exp
                # Bullet repayment in final year
                debt_repayment = debt_amt if t == lifetime - 1 else 0

            ebit = revenue - opex - depreciation_t
            ebt = ebit - interest_t
            tax = max(ebt, 0) * (tax_rate / 100)  # Tax cannot be negative
            net_income = ebt - tax

            # FCFE = Net income + Depreciation - CapEx - Debt Repayment + Net Borrowing (NWC ignored)
            fcfe = net_income + depreciation_t - capex - debt_repayment + net_borrowing

            # Discounted FCFE for NPV
            disc_fcfe = fcfe / ((1 + cost_of_equity_rate) ** t)

            # Store all steps for table
            year_list.append(t)
            eff_MWh_list.append(eff_MWh)
            revenue_list.append(revenue)
            opex_list.append(opex)
            capex_list.append(capex)
            depreciation_list.append(depreciation_t)
            interest_list.append(interest_t)
            ebit_list.append(ebit)
            ebt_list.append(ebt)
            tax_list.append(tax)
            net_income_list.append(net_income)
            debt_repay_list.append(debt_repayment)
            net_borrow_list.append(net_borrowing)
            fcfe_list.append(fcfe)
            disc_fcfe_list.append(disc_fcfe)

        # Cumulative Discounted FCFE
        cumulative_fcfe = np.cumsum(disc_fcfe_list)

        # --- Profitability Metrics ---
        equity_npv = np.sum(disc_fcfe_list)
        mirr = calculate_mirr(fcfe_list, finance_rate=cost_of_debt/100, reinvest_rate=cost_of_equity/100)
        payback = calculate_payback_period(fcfe_list)
        pi = (np.sum(disc_fcfe_list[1:])) / abs(fcfe_list[0])  # exclude initial investment in numerator
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("NPV (Equity, discounted @ Cost of Equity)", f"€{equity_npv:,.0f}")
        with col2:
            st.metric("Cost of Equity", f"{cost_of_equity:.2f}%")
        with col3:
            if np.isnan(mirr):
                st.metric("MIRR (Equity)", "N/A")
            else:
                st.metric("MIRR (Equity)", f"{mirr*100:.2f}%")
        with col4:
            if np.isnan(payback):
                st.metric("Payback Period", "Never")
            else:
                st.metric("Payback Period (years)", payback)
        with col5:
            st.metric("Profitability Index", f"{pi:.2f}")

        # --- DCF Chart (top) ---
        fcfe_df = pd.DataFrame({
            "Year": year_list,
            "Discounted FCFE (€)": disc_fcfe_list,
            "Cumulative Discounted FCFE (€)": cumulative_fcfe
        })
        fcfe_chart = (
            alt.Chart(fcfe_df)
            .mark_bar(color="orange")
            .encode(x="Year:O", y=alt.Y("Discounted FCFE (€):Q", title="Discounted FCFE (€)"))
            +
            alt.Chart(fcfe_df)
            .mark_line(point=True, color="blue")
            .encode(x="Year:O", y=alt.Y("Cumulative Discounted FCFE (€):Q", title="Cumulative Discounted FCFE (€)", axis=alt.Axis(orient="right")))
        )
        st.altair_chart(fcfe_chart.resolve_scale(y="independent").properties(height=350, title="Equity Cashflow DCF"), use_container_width=True)

        # --- Table (middle) ---
        table_df = pd.DataFrame({
            "Year": year_list,
            "Eff. MWh": eff_MWh_list,
            "Revenue (€)": revenue_list,
            "O&M (€)": opex_list,
            "CapEx (€)": capex_list,
            "Depreciation (€)": depreciation_list,
            "EBIT (€)": ebit_list,
            "Interest Expense (€)": interest_list,
            "EBT (€)": ebt_list,
            "Tax (€)": tax_list,
            "Net Income (€)": net_income_list,
            "Debt Repayment (€)": debt_repay_list,
            "Net Borrowing (€)": net_borrow_list,
            "FCFE (€)": fcfe_list,
            "Discounted FCFE (€)": disc_fcfe_list,
            "Cumulative Discounted FCFE (€)": cumulative_fcfe
        })
        st.dataframe(table_df)

        # --- Download as Excel ---
        excel_buffer = BytesIO()
        table_df.to_excel(excel_buffer, index=False, engine="xlsxwriter")
        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name="fcfe_cashflows.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # --- Sensitivity: NPV vs Cost of Equity (bottom of outputs) ---
        st.markdown("### Sensitivity: NPV vs Cost of Equity (%)")
        coe_range = np.linspace(0.01, 0.18, 30)
        npvs = []
        for coe in coe_range:
            disc_fcfe = [fcfe_list[t] / ((1+coe)**t) for t in range(lifetime)]
            npvs.append(np.sum(disc_fcfe))
        sens_df = pd.DataFrame({"Cost of Equity (%)": coe_range*100, "NPV (€)": npvs})
        sens_chart = (
            alt.Chart(sens_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Cost of Equity (%):Q", title="Cost of Equity (%)"),
                y=alt.Y("NPV (€):Q", title="NPV (€)")
            )
            .properties(height=350, title="NPV Sensitivity to Cost of Equity")
        )
        st.altair_chart(sens_chart, use_container_width=True)

        # ---- Financing Structure Pie Chart ----
        st.markdown("### Financing Structure")
        pie_labels = ["Debt", "Equity"]
        pie_values = [debt_amt, equity_amt]
        fig = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, hole=0.45, marker=dict(line=dict(color='#FFF', width=2)))])
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- Energy-Price Forecasts Chart (from Energy Prices tab) ---
    st.markdown("## Energy-Price Forecasts (2024–2043)")
    years = [str(2024 + t) for t in range(20)]
    price_df = (
        price_scenarios_df[years]
         .reset_index()
         .melt(id_vars="Scenario", var_name="Year", value_name="Price")
    )
    price_df["YearIndex"] = price_df["Year"].astype(int) - 2024
    chart = (
        alt.Chart(price_df).mark_line(point=True)
           .encode(x="YearIndex:O", y="Price:Q", color="Scenario:N")
           .properties(height=300, title="Energy Prices")
    )
    st.altair_chart(chart, use_container_width=True)

    # --- Explanations (after all outputs) ---
    st.markdown("""
    ### How is the expected MWh calculated?

    The **expected annual energy production (MWh)** is estimated as:
    $$
    P = 0.5 \\times \\rho \\times A \\times v^3 \\times C_p
    $$
    where:  
    - $\\rho$ = average air density at site (kg/m³)  
    - $A$ = swept rotor area (m²), $= \\pi \\times (\\text{Rotor Diameter}/2)^2$  
    - $v$ = average wind speed at hub height (m/s)  
    - $C_p$ = power coefficient (turbine-specific)  
    - $0.5 \\times \\rho \\times A \\times v^3 \\times C_p$ gives the average power (Watt) per turbine.  
    - Multiply by number of turbines, operational hours (8760/year), **availability**, and any other factors for site.

    _This dashboard estimates output for your selected site polygon using site wind/air rasters, selected turbine specs, and operational factors like ramp-up, degradation, and availability._
    """)

    st.markdown("""
    ### Free Cash Flow to Equity (FCFE) Formula and Explanation

    $$
    \\text{FCFE} = \\text{Net Income} + \\text{Depreciation} - \\text{CapEx} - \\text{Debt Repayment} + \\text{Net Borrowing}
    $$

    **Where:**
    - **Net Income**: After-tax profit after deducting interest expense.
    - **Depreciation**: Non-cash charge (here, straight-line).
    - **CapEx**: Capital Expenditures (here, only in Year 0).
    - **Debt Repayment**: Repayment of principal (here, "bullet" in final year).
    - **Net Borrowing**: New debt raised (here, only in Year 0).
    - *No NWC modeled (set to 0 for simplicity).*
    """)



# --- 5) News Tab (Google News RSS as real news feed) ---
with news_tab:
    st.markdown("## 📰 Latest Wind & Renewable Subsidy News (Germany/EU)")
    FEED_URL = "https://news.google.com/rss/search?q=wind+energy+subsidy+germany+OR+eu+OR+renewable+subsidy&hl=en&gl=DE&ceid=DE:en"
    try:
        feed = feedparser.parse(FEED_URL)
        if hasattr(feed, "entries") and len(feed.entries) > 0:
            st.info("Live headlines from Google News (most recent first):")
            for entry in feed.entries[:10]:
                # A more 'news-card' look
                st.markdown(
                    f"""
                    <div style="background-color:#f9f9f9; border-radius:8px; margin-bottom:16px; padding:14px; border:1px solid #eee;">
                        <div style="display:flex; align-items:center;">
                            <span style="font-size:1.5em; margin-right:12px;">🗞️</span>
                            <span style="font-weight:bold; font-size:1.12em;">{entry.title}</span>
                        </div>
                        <div style="color:#444; margin-top:5px; margin-bottom:9px;">
                            {entry.summary if 'summary' in entry else ''}
                        </div>
                        <a href="{entry.link}" target="_blank" style="color:#2257d2; text-decoration: none; font-weight:bold;">Read more →</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No news available at the moment. Try again soon.")
    except Exception as e:
        st.error(f"Error fetching news: {e}")

 # --- 6) LCOE Calculator Tab ---
with lcoe_tab:
    st.markdown("## 🧮 Levelized Cost of Energy (LCOE) Calculator")
    st.write("This tool estimates LCOE for your selected site and compares to future energy prices.")

    if not st.session_state.get("st_data", {}).get("last_active_drawing"):
        st.info("Draw a site polygon on the Wind Map first.")
    else:
        poly = st.session_state["st_data"]["last_active_drawing"]["geometry"]

        def extract_mean(poly_geojson, rast):
            geom = shape(poly_geojson)
            gdf = gpd.GeoDataFrame(
                geometry=[geom], crs="EPSG:4326"
            ).to_crs(rast.crs)
            arr, _ = mask(rast, [mapping(gdf.geometry.iloc[0])], crop=True)
            vals = arr[0][arr[0] > 0]
            return float(np.nanmean(vals)) if vals.size else 0.0

        with rasterio.open(wind_path) as ws:
            avg_wind = extract_mean(poly, ws)
        with rasterio.open(air_path) as ad:
            avg_air = extract_mean(poly, ad)

        D, Cp = turbine["rotor_diameter_m"], turbine["cp"]
        P_W     = 0.5 * avg_air * np.pi*(D/2)**2 * (avg_wind**3) * Cp
        P_MW    = P_W / 1e6
        cap_mw  = turbine["rated_power_kw"] / 1000
        avg_P   = min(P_MW, cap_mw)
        total_MW, total_MWh = avg_P * n_turbines, avg_P * 8760 * n_turbines

        availability     = 0.95
        ramp_up          = [0.75] + [1.0]*19
        degradation      = [(1 - 0.005*t) for t in range(20)]
        lifetime = 20
        omc = om_cost
        inflation = inflation_rate / 100

        # Calculate discounted sum of all costs (CAPEX, OPEX), discounted sum of all MWh
        disc_rate = cost_of_equity / 100
        total_disc_cost = 0
        total_disc_mwh  = 0
        for t in range(lifetime):
            if t == 0:
                cost = capex_eur
                mwh = 0
            else:
                cost = omc * total_MW * ((1 + inflation)**(t-1))
                mwh  = total_MWh * availability * ramp_up[t] * degradation[t]
            total_disc_cost += cost / ((1 + disc_rate) ** t)
            total_disc_mwh  += mwh  / ((1 + disc_rate) ** t)
        lcoe = total_disc_cost / total_disc_mwh if total_disc_mwh > 0 else np.nan

        st.metric("LCOE (€/MWh)", f"{lcoe:,.2f}")

        # Compare LCOE with future energy prices
        years = [str(2024 + t) for t in range(lifetime)]
        price_series = []
        for t in range(lifetime):
            y = years[t] if years[t] in price_scenarios_df.columns else years[-1]
            price_series.append(price_scenarios_df.at[scenario, y])
        df = pd.DataFrame({
            "Year": [2024 + t for t in range(lifetime)],
            "LCOE (€/MWh)": [lcoe]*lifetime,
            "Price (€/MWh)": price_series
        })
        chart = alt.Chart(df).transform_fold(
            ["LCOE (€/MWh)", "Price (€/MWh)"],
            as_=["Metric", "Value"]
        ).mark_line(point=True).encode(
            x="Year:O",
            y="Value:Q",
            color="Metric:N"
        ).properties(height=350, title="LCOE vs. Market Price")
        st.altair_chart(chart, use_container_width=True)


# --- 7) Site Score Table Tab ---
with table_tab:
    st.markdown("## 📄 All Site Scores by Location")

    # Re-load the correct score raster for extraction (full resolution, all cells)
    SCORE_FILE_ID = "17EZxRok4zRmS7iX1Y1aR7Znh2RpnvLmd"
    score_path = gdrive_download(SCORE_FILE_ID, ".tif")
    with rasterio.open(score_path) as score_src:
        score_data = np.nan_to_num(score_src.read(1), nan=0.0, posinf=0.0, neginf=0.0)
        score_transform = score_src.transform

    coords = []
    values = []
    rows, cols = score_data.shape
    # Show full raster shape and cell count
    for row in range(rows):
        for col in range(cols):
            score = score_data[row, col]
            x, y = rasterio.transform.xy(score_transform, row, col, offset='center')
            coords.append((y, x))  # lat, lon
            values.append(score)

    score_df = pd.DataFrame(coords, columns=["Latitude", "Longitude"])
    score_df["Site Score"] = values
    score_df = score_df.sort_values("Site Score", ascending=False)
    # Warn if table is very large
    if len(score_df) > 100_000:
        st.warning(f"Warning: Displaying all {len(score_df):,} site scores may be slow in your browser.")
        st.dataframe(score_df.head(100_000), use_container_width=True)
        st.caption("Showing first 100,000 rows only.")
    else:
        st.dataframe(score_df, use_container_width=True)
