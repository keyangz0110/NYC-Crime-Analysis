import pandas as pd
from geoprocessing import map_to_neighborhoods
from crime_classifier import train_violent_crime_classifier

# Load data
print("Loading arrest data...")
df = pd.read_csv('data/cleaned_data.csv', low_memory=False)

# Print some information about the loaded data
print(f"Loaded {len(df)} records")
print(f"Available columns: {df.columns.tolist()}")

# Check if neighborhood data already exists
if 'NEIGHBORHOOD' not in df.columns:
    print("Mapping arrests to neighborhoods...")
    df_with_neighborhoods = map_to_neighborhoods(df)
else:
    print("Neighborhood data already present in dataset")
    df_with_neighborhoods = df

# Train classifier
model_package = train_violent_crime_classifier(df_with_neighborhoods)

print("Violent crime classifier training complete!")
