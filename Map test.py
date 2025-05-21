import streamlit as st
import folium
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, Polygon, Point
from branca.colormap import LinearColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tempfile import NamedTemporaryFile
from streamlit_folium import st_folium
import random

st.set_page_config(layout="wide")
st.title("🌬️ Germany Wind Speed Map (Pixel-Precise Tooltips)")

# --- Upload GeoTIFF ---
uploaded_tif = st.file_uploader("Upload a GeoTIFF wind speed file", type=["tif", "tiff"])
if not uploaded_tif:
    st.info("Upload a GeoTIFF file, such as 'simulated_germany_wind_speed.tif'")
    st.stop()

with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
    tmp.write(uploaded_tif.read())
    tif_path = tmp.name

# --- Load and mask raster using Germany boundary ---
@st.cache_resource
def load_wind_data(tif_path):
    germany = gpd.read_file("https://raw.githubusercontent.com/johan/world.geo.json/master/countries/DEU.geo.json")
    with rasterio.open(tif_path) as src:
        germany = germany.to_crs(src.crs)
        out_image, out_transform = mask(src, germany.geometry.apply(mapping), crop=True)
        wind_array = out_image[0]
        raster_bounds = src.bounds
    return wind_array, out_transform, raster_bounds, germany

wind_data, transform, bounds, germany = load_wind_data(tif_path)

# --- Create overlay image with boosted viridis contrast ---
@st.cache_resource
def create_overlay(wind_data):
    vmin, vmax = np.nanmin(wind_data), np.nanmax(wind_data)
    norm = np.clip((wind_data - vmin) / (vmax - vmin), 0, 1) ** 0.8  # boost contrast
    cmap = plt.get_cmap("viridis")
    rgba = cmap(norm)
    rgba[..., 3] = (wind_data > 0).astype(float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgba, origin='upper')
    ax.axis('off')

    with NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        plt.savefig(tmp_img.name, bbox_inches='tight', pad_inches=0, dpi=150, transparent=True)
        plt.close()
        return tmp_img.name, float(vmin), float(vmax)

img_path, vmin, vmax = create_overlay(wind_data)

# --- Create light map with labels clearly visible
center = [germany.geometry.centroid.y.mean(), germany.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=6, tiles="CartoDB.Positron")

# Raster bounds
h, w = wind_data.shape
x_min, y_max = transform[2], transform[5]
x_res, y_res = transform[0], transform[4]
x_max = x_min + x_res * w
y_min = y_max + y_res * h
bounds_latlon = [[y_min, x_min], [y_max, x_max]]

# ✅ Overlay with opacity to allow label visibility
folium.raster_layers.ImageOverlay(
    name="Wind Speed Overlay",
    image=img_path,
    bounds=bounds_latlon,
    opacity=0.4
).add_to(m)

# Colormap
viridis = cm.get_cmap("viridis", 256)
colors = [mcolors.rgb2hex(viridis(i)) for i in range(viridis.N)]
colormap = LinearColormap(colors, vmin=vmin, vmax=vmax)
colormap.caption = "Wind Speed (m/s)"
colormap.add_to(m)

# Germany border
germany_union = germany.geometry.unary_union
folium.GeoJson(
    germany_union,
    name="Germany Border",
    style_function=lambda x: {"color": "red", "weight": 2, "fillOpacity": 0}
).add_to(m)

# ✅ Precise hoverable rectangles (1 per raster cell)
step = 1
for i in range(0, wind_data.shape[0] - step, step):
    for j in range(0, wind_data.shape[1] - step, step):
        val = wind_data[i, j]
        if not np.isnan(val) and val > 0:
            lat_top = float(transform[5] + i * y_res)
            lon_left = float(transform[2] + j * x_res)
            lat_bot = lat_top + step * y_res
            lon_right = lon_left + step * x_res
            folium.Rectangle(
                bounds=[[lat_bot, lon_left], [lat_top, lon_right]],
                color=None,
                fill=True,
                fill_color="#00000000",
                fill_opacity=0.01,
                tooltip=folium.Tooltip(f"{val:.2f} m/s", sticky=True)
            ).add_to(m)

# ✅ Add restricted zones randomly inside Germany
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

# ✅ Power grid lines (interconnected and inside Germany)
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

# --- Display map ---
st_folium(m, width=1000, height=700)