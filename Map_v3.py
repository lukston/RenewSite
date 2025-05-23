import streamlit as st
import folium
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterio.transform import array_bounds
from shapely.geometry import mapping, Point
from branca.colormap import LinearColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tempfile import NamedTemporaryFile
from streamlit_folium import st_folium
import requests
import random

st.set_page_config(layout="wide")
st.title("🎯 Germany Score Map (0–100 Scale)")

# --- Auto-Download GeoTIFF from GitHub ---
@st.cache_resource
def download_and_generate_score_tif():
    url = "https://github.com/lukston/RenewSite/blob/main/simulated_germany_wind_speed.tif"  # ✅ <--- HIER ANPASSEN

    response = requests.get(url)
    response.raise_for_status()

    # Save the original file temporarily
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_in:
        tmp_in.write(response.content)
        tmp_in_path = tmp_in.name

    # Load and generate score overlay
    with rasterio.open(tmp_in_path) as src:
        profile = src.profile.copy()
        shape = src.read(1).shape
        transform = src.transform

    scores = np.random.randint(0, 101, size=shape).astype(np.float32)

    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_out:
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(tmp_out.name, "w", **profile) as dst:
            dst.write(scores, 1)
        return tmp_out.name

# Download and score
tif_path = download_and_generate_score_tif()

# --- Load and mask raster with Germany border ---
@st.cache_resource
def load_score_data(tif_path):
    germany = gpd.read_file("https://raw.githubusercontent.com/johan/world.geo.json/master/countries/DEU.geo.json")
    with rasterio.open(tif_path) as src:
        germany = germany.to_crs(src.crs)
        out_image, out_transform = mask(src, germany.geometry.apply(mapping), crop=True)
        score_array = out_image[0]
    return score_array, out_transform, germany

score_data, transform, germany = load_score_data(tif_path)

# --- Create overlay image ---
@st.cache_resource
def create_overlay(score_data):
    vmin, vmax = 0, 100
    norm = np.clip((score_data - vmin) / (vmax - vmin), 0, 1) ** 0.8
    cmap = plt.get_cmap("RdYlGn")
    rgba = cmap(norm)
    rgba[..., 3] = (score_data >= 0).astype(float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgba, origin='upper')
    ax.axis('off')

    with NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        plt.savefig(tmp_img.name, bbox_inches='tight', pad_inches=0, dpi=150, transparent=True)
        plt.close()
        return tmp_img.name, float(vmin), float(vmax)

img_path, vmin, vmax = create_overlay(score_data)

# --- Map ---
center = [germany.geometry.centroid.y.mean(), germany.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=6, tiles="CartoDB.Positron")

# Bounds basierend auf maskierten Daten
h, w = score_data.shape
ymin, ymax, xmin, xmax = array_bounds(h, w, transform)
bounds_latlon = [[ymin, xmin], [ymax, xmax]]

folium.raster_layers.ImageOverlay(
    name="Score Overlay",
    image=img_path,
    bounds=bounds_latlon,
    opacity=0.5
).add_to(m)

colors = [mcolors.rgb2hex(cm.get_cmap("RdYlGn")(i)) for i in range(256)]
colormap = LinearColormap(colors, vmin=vmin, vmax=vmax)
colormap.caption = "Score (0–100)"
colormap.add_to(m)

germany_union = germany.geometry.unary_union
folium.GeoJson(
    germany_union,
    name="Germany Border",
    style_function=lambda x: {"color": "red", "weight": 2, "fillOpacity": 0}
).add_to(m)

# Hover rectangles
step = 1
for i in range(0, score_data.shape[0] - step, step):
    for j in range(0, score_data.shape[1] - step, step):
        val = score_data[i, j]
        if not np.isnan(val) and val >= 0:
            lat_top = float(transform[5] + i * transform[4])
            lon_left = float(transform[2] + j * transform[0])
            lat_bot = lat_top + step * transform[4]
            lon_right = lon_left + step * transform[0]
            folium.Rectangle(
                bounds=[[lat_bot, lon_left], [lat_top, lon_right]],
                color=None,
                fill=True,
                fill_color="#00000000",
                fill_opacity=0.01,
                tooltip=folium.Tooltip(f"{val:.0f} / 100", sticky=True)
            ).add_to(m)

# Restricted zones
random.seed(42)
restricted_areas = []
germany_bounds = germany_union.bounds

while len(restricted_areas) < 10:
    lon = random.uniform(germany_bounds[0], germany_bounds[2])
    lat = random.uniform(germany_bounds[1], germany_bounds[3])
    pt = Point(lon, lat)
    if germany_union.contains(pt):
        size = 0.2
        restricted_areas.append({
            "name": f"Restricted Zone {len(restricted_areas)+1}",
            "coordinates": [[
                [lat - size, lon - size],
                [lat - size, lon + size],
                [lat + size, lon + size],
                [lat + size, lon - size],
                [lat - size, lon - size]
            ]]
        })

for area in restricted_areas:
    folium.Polygon(
        locations=area["coordinates"],
        color="red",
        fill=True,
        fill_opacity=0.4,
        tooltip=area["name"]
    ).add_to(m)

# Power grid lines
power_nodes = [
    [53.0, 9.5],
    [52.0, 10.5],
    [51.0, 9.5],
    [50.0, 10.0],
    [49.0, 9.0],
    [48.0, 10.5],
    [47.5, 9.0],
]

connections = [
    (0, 1), (1, 2), (2, 3),
    (3, 4), (4, 5), (5, 6),
    (1, 3), (2, 4), (3, 5)
]

for i, j in connections:
    folium.PolyLine(
        locations=[power_nodes[i], power_nodes[j]],
        color="yellow",
        weight=3,
        opacity=0.9,
        tooltip="Power Line"
    ).add_to(m)

# Display map
st_folium(m, width=1000, height=700)