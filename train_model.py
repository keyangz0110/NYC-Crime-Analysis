import pandas as pd
import pickle
from prophet import Prophet
from model_utils import build_forecast_model

# Load your data
df = pd.read_csv('data/cleaned_data.csv')

# Convert dates to datetime format
df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'], format='%Y/%m/%d')

# Aggregate data by date to create daily counts
daily_arrests = df.groupby(df['ARREST_DATE'].dt.date).size().reset_index()
daily_arrests.columns = ['ARREST_DATE', 'arrest_count']

# Convert date back to datetime (needed for Prophet)
daily_arrests['ARREST_DATE'] = pd.to_datetime(daily_arrests['ARREST_DATE'])

# Print data to verify
print(f"Created daily aggregation with {len(daily_arrests)} rows")
print(daily_arrests.head())

# Train the model
model = build_forecast_model(daily_arrests)

# Save the model
with open('arrest_forecast_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as 'arrest_forecast_model.pkl'")
