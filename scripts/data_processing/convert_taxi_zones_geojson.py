from pathlib import Path

import geopandas as gpd

INPUT_SHP = "data/raw/taxi_zones/taxi_zones.shp"
OUTPUT_GEOJSON = "frontend/public/data/taxi_zones.geojson"

def main() -> None:
    shp_path = Path(INPUT_SHP)
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(epsg=4326)

    # Normalize zone id naming so it matches forecast rows directly.
    if "locationid" in gdf.columns and "PULocationID" not in gdf.columns:
        gdf["PULocationID"] = gdf["locationid"].astype(int)
    elif "LocationID" in gdf.columns and "PULocationID" not in gdf.columns:
        gdf["PULocationID"] = gdf["LocationID"].astype(int)

    keep_cols = [c for c in ["PULocationID", "borough", "zone", "service_zone"] if c in gdf.columns]
    gdf = gdf[keep_cols + ["geometry"]]

    out_path = Path(OUTPUT_GEOJSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")

    print("saved:", out_path)
    print("rows:", len(gdf))
    print("columns:", list(gdf.columns))


if __name__ == "__main__":
    main()
