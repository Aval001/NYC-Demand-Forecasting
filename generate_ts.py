import os
import pandas as pd
from calendar import monthrange
import geopandas as gpd

zones = gpd.read_file('data/network_shapefile/geo_export.shp')
borough_zones_raw = zones[zones.borough == 'Manhattan']
borough_zones = borough_zones_raw[borough_zones_raw['location_i'] != 103]

zone_ids = borough_zones['location_i'].values 

# Parameters
input_folder = "data/with_zones"
output_folder = "data/processed"
os.makedirs(output_folder, exist_ok=True)

file_names = [f"output_{i}.csv" for i in range(1, 8)]
resolution = "5min"

# Time series generation
for i, file in enumerate(file_names, start=1):
    file_path = os.path.join(input_folder, file)
    df1 = pd.read_csv(file_path,low_memory=False)

    df1.columns = df1.columns.str.strip()

    # Drop missing pickup zones and enforce integer type
    df1 = df1.dropna(subset=["pickup_zone"])
    df1["pickup_zone"] = df1["pickup_zone"].astype(int)

    # Filter for valid zone IDs only
    df1 = df1[df1["pickup_zone"].isin(zone_ids)]

    # Sort and bin timestamps
    df1["pickup_datetime"] = pd.to_datetime(df1["pickup_datetime"], errors="coerce")
    df1 = df1.dropna(subset=["pickup_datetime"])  # Drop rows where conversion failed
    df1.sort_values("pickup_datetime", inplace=True)
    df1["pickup_time_bin"] = df1["pickup_datetime"].dt.floor(resolution)

    # Group by time bin and pickup zone
    grouped = df1.groupby(["pickup_time_bin", "pickup_zone"]).size().unstack(fill_value=0)

    # Reindex columns to match all zone_ids
    grouped = grouped.reindex(columns=zone_ids, fill_value=0)

    # Define full time range for the month
    year = 2010
    month = i
    days_in_month = monthrange(year, month)[1]
    start = pd.Timestamp(f"{year}-{month:02d}-01 00:00:00")
    end = pd.Timestamp(f"{year}-{month:02d}-{days_in_month} 23:55:00")
    full_time_index = pd.date_range(start=start, end=end, freq=resolution)

    # Reindex time index to full month
    grouped = grouped.reindex(full_time_index, fill_value=0)

    # Save to file
    output_path = os.path.join(output_folder, f"processed_{i:02d}.csv")
    grouped.to_csv(output_path)


