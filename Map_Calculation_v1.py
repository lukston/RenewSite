from rasterio.features import rasterize
import os
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape

wind_path = "/Users/lukassteinseifer/Documents/Studium/FIEP/Data_RenewSite/Input_Data/DEU_wind-speed_50m.tif"
air_path = "/Users/lukassteinseifer/Documents/Studium/FIEP/Data_RenewSite/Input_Data/DEU_air-density_50m.tif"
restricted_path = "/Users/lukassteinseifer/Documents/Studium/FIEP/Data_RenewSite/Input_Data/georef-germany-gemeinde@public.geojson"
output_path = "/Users/lukassteinseifer/Documents/Studium/FIEP/Data_RenewSite/Output_Data/precalculated_site_score.tif"

# Ensure output folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Validate input files
assert os.path.exists(wind_path), "Wind raster not found"
assert os.path.exists(air_path), "Air raster not found"
assert os.path.exists(restricted_path), "Restricted GeoJSON not found"

with rasterio.open(wind_path) as wind_src, rasterio.open(air_path) as air_src:
    wind = wind_src.read(1)
    air = air_src.read(1)
    profile = wind_src.profile
    transform = wind_src.transform
    crs = wind_src.crs

site_score = wind * air
print("Wind stats:", np.min(wind), np.max(wind))
print("Air stats:", np.min(air), np.max(air))
print("Score (before mask):", np.min(site_score), np.max(site_score))
restricted = gpd.read_file(restricted_path).to_crs(crs)
restricted = restricted.to_crs(restricted.estimate_utm_crs())
restricted["geometry"] = restricted.centroid.buffer(2000)
restricted = restricted.to_crs(crs)
shapes = [(geom, 1) for geom in restricted.geometry]
mask = rasterize(shapes, out_shape=site_score.shape, transform=transform, fill=0, dtype="uint8")
print("Restricted area coverage (pixels):", np.sum(mask))
site_score[mask == 1] = 0
print("Score (after mask):", np.min(site_score), np.max(site_score))

profile.update(dtype="float32", count=1, compress="lzw")
with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(site_score.astype("float32"), 1)
    print("✅ Site score written to:", output_path)