import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def add_zones_vectorized(df, borough_zones):
    # Create GeoDataFrames for pickup and dropoff points
    gdf_pickup = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[' pickup_longitude'], df[' pickup_latitude']),
        crs=borough_zones.crs
    )

    gdf_dropoff = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[' dropoff_longitude'], df[' dropoff_latitude']),
        crs=borough_zones.crs
    )

    # Spatial join pickup points to zones
    pickup_join = gpd.sjoin(gdf_pickup, borough_zones[['geometry', 'location_i']], how='left', predicate='within')
    df[' pickup_zone'] = pickup_join['location_i'].fillna('nan').values

    # Spatial join dropoff points to zones
    dropoff_join = gpd.sjoin(gdf_dropoff, borough_zones[['geometry', 'location_i']], how='left', predicate='within')
    df[' dropoff_zone'] = dropoff_join['location_i'].fillna('nan').values

    # Convert datetime columns
    df[' pickup_datetime'] = pd.to_datetime(df[' pickup_datetime'])
    df[' dropoff_datetime'] = pd.to_datetime(df[' dropoff_datetime'])

    # Drop the temporary geometry column if needed
    df.drop(columns='geometry', inplace=True, errors='ignore')

    return df

# Load borough zones once
zones = gpd.read_file('data/network_shapefile/geo_export.shp')
borough_zones_raw = zones[zones.borough == 'Manhattan']
borough_zones = borough_zones_raw[borough_zones_raw['location_i'] != 103]

# Loop through files 1 to 12
for i in range(1,8):
    file_path = f'data/raw/trip_data_{i}.csv'
    output_path = f'data/with_zones/output_{i:02d}.csv'

    print(f"Processing file: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)  # avoid dtype warning
    df = add_zones_vectorized(df, borough_zones)
    df.to_csv(output_path, index=False)

print("All files processed.")
