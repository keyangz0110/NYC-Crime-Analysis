import pandas as pd
import pickle
from prophet import Prophet
from model_utils import build_forecast_model
import os

# Display some information about the process
print("Starting model training process...")

# Path to the new merged data file
data_file = 'data/cleaned_data.csv'
print(f"Loading data from {data_file}...")

# Load your data - specify low_memory=False for large files
df = pd.read_csv(data_file, low_memory=False)
print(f"Loaded {len(df)} records")

# Convert dates to datetime format with the new yyyy-mm-dd format
print("Processing dates...")
df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'], format='%Y-%m-%d')

# Check date range to confirm data looks good
print(f"Data spans from {df['ARREST_DATE'].min()} to {df['ARREST_DATE'].max()}")

# Aggregate data by date to create daily counts
print("Aggregating data by date...")
daily_arrests = df.groupby(df['ARREST_DATE'].dt.date).size().reset_index()
daily_arrests.columns = ['ARREST_DATE', 'arrest_count']

# Convert date back to datetime (needed for Prophet)
daily_arrests['ARREST_DATE'] = pd.to_datetime(daily_arrests['ARREST_DATE'])

# Save the aggregated data for reference
daily_arrests.to_csv('data/arrest_daily.csv', index=False)
print(f"Created daily aggregation with {len(daily_arrests)} rows")
print(daily_arrests.head())

# Train the model
print("Training the forecasting model (this may take a few minutes)...")
model = build_forecast_model(daily_arrests)

# Save the model
model_file = 'arrest_forecast_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved as '{model_file}'")
print(f"File size: {os.path.getsize(model_file) / (1024*1024):.2f} MB")
