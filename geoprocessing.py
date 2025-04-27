import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def map_to_neighborhoods(df):
    # Load NYC neighborhood boundaries 
    # Source: https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd/about_data
    nyc_neighborhoods = gpd.read_file('2020_neighborhood_tabulation_areas.geojson')
    
    # Convert lat/long to geometry points
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Spatial join with neighborhoods
    neighborhood_data = gpd.sjoin(geo_df, nyc_neighborhoods, how="left", predicate='within')
    
    # Clean up the result
    df_with_neighborhoods = pd.DataFrame(neighborhood_data)
    df_with_neighborhoods.rename(columns={'ntaname': 'NEIGHBORHOOD'}, inplace=True)
    
    print(f"Mapped {len(df_with_neighborhoods[~df_with_neighborhoods['NEIGHBORHOOD'].isna()])} records to neighborhoods")
    
    return df_with_neighborhoods

# Usage in train_model.py
# df = map_to_neighborhoods(df)
