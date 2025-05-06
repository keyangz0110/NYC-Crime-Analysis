# Import libraries
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from prophet.diagnostics import cross_validation, performance_metrics

# Model building function
def build_forecast_model(df):
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    df_prophet = df.rename(columns={'ARREST_DATE': 'ds', 'arrest_count': 'y'})
    
    # Initialize and train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    # Add US holidays for better forecasting
    model.add_country_holidays(country_name='US')
    
    # Fit model
    model.fit(df_prophet)
    
    return model

# Prediction function
def make_forecast(model, periods=30):
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Make prediction
    forecast = model.predict(future)
    
    return forecast

def tune_model(df, param_grid):
    best_mae = float('inf')
    best_params = None
    
    for changepoint_prior in param_grid['changepoint_prior_scale']:
        for seasonality_prior in param_grid['seasonality_prior_scale']:
            for holidays_prior in param_grid['holidays_prior_scale']:
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior,
                    seasonality_prior_scale=seasonality_prior,
                    holidays_prior_scale=holidays_prior,
                    seasonality_mode='multiplicative'
                )
                
                model.fit(df)
                
                # Cross-validation
                df_cv = cross_validation(model, initial='180 days', period='30 days', horizon='90 days')
                df_p = performance_metrics(df_cv)
                
                # Check if this model is better
                if df_p['mae'].mean() < best_mae:
                    best_mae = df_p['mae'].mean()
                    best_params = {
                        'changepoint_prior_scale': changepoint_prior,
                        'seasonality_prior_scale': seasonality_prior,
                        'holidays_prior_scale': holidays_prior
                    }
    
    return best_params
