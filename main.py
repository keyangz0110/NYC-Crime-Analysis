import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import pickle
import plotly.graph_objects as go
from geoprocessing import map_to_neighborhoods
from crime_classifier import train_violent_crime_classifier, map_to_violent_category
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
from datetime import datetime

# Load data
df_monthly = pd.read_csv("data/arrest_monthly.csv", names=["month", "arrest_count"], header=None)
df_daily = pd.read_csv("data/arrest_daily.csv", names=["day", "arrest_count"], header=None)
df_top = pd.read_csv("data/arrest_top_offenses.csv", names=["offense", "arrest_count"], header=None)
df_demo = pd.read_csv("data/arrest_demographics.csv", names=["sex", "age_group", "race", "arrest_count"], header=None)
# Sex Map
sex_map = {
    "M": "Male",
    "F": "Female",
    "U": "Unknown"
}
# Borough Map
boro_map = {
    "K": "Brooklyn",
    "B": "Bronx",
    "M": "Manhattan",
    "Q": "Queens",
    "S": "Staten Island"
}
# Age Map
age_map = {
    "<18": "Under 18",
    "18-24": "18 to 24",
    "25-44": "25 to 44",
    "45-64": "45 to 64",
    "65+": "65 and above",
    "UNKNOWN": "Unknown"
}
# Borough × Gender Map
df_boro_sex = pd.read_csv("data/arrest_borough_gender.csv", names=["ARREST_BORO", "PERP_SEX", "arrest_count"], header=None)
df_boro_sex["ARREST_BORO"] = df_boro_sex["ARREST_BORO"].map(boro_map)
df_boro_sex["PERP_SEX"] = df_boro_sex["PERP_SEX"].map(sex_map)
# Borough × Age Map
df_boro_age = pd.read_csv("data/arrest_borough_age.csv", names=["ARREST_BORO", "AGE_GROUP", "arrest_count"], header=None)
df_boro_age["ARREST_BORO"] = df_boro_age["ARREST_BORO"].map(boro_map)
df_boro_age["AGE_GROUP"] = df_boro_age["AGE_GROUP"].map(age_map)
# Borough × Race Map
df_boro_race = pd.read_csv("data/arrest_borough_race.csv", names=["ARREST_BORO", "PERP_RACE", "arrest_count"], header=None)
df_boro_race["ARREST_BORO"] = df_boro_race["ARREST_BORO"].map(boro_map)

df_map = pd.read_csv("data/cleaned_data.csv")
# Date format
df_map['ARREST_DATE'] = pd.to_datetime(df_map['ARREST_DATE'], format='%Y-%m-%d')

# Convert time to Python native datetime object
min_date = df_map['ARREST_DATE'].min().to_pydatetime()
max_date = df_map['ARREST_DATE'].max().to_pydatetime()


st.set_page_config(layout="wide")
st.title("🚔 NYC Arrests Interactive Dashboard")

# Tabs
tabs = st.tabs(["📈 Trends", "👤 Demographics", "🧭 Borough Comparison", "🗺️ Maps", "🔝 Top Offenses", "🔮 Arrest Forecast", "🔍 Crime Predictor"])

# Trends Tab
with tabs[0]:
    st.subheader("Monthly Arrest Trends")
    fig_month = px.line(df_monthly, x="month", y="arrest_count", markers=True)
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Daily Arrest Trends")
    fig_day = px.line(df_daily, x="day", y="arrest_count")
    st.plotly_chart(fig_day, use_container_width=True)

# Demographics Tab
with tabs[1]:
    st.subheader("Gender Distribution")
    fig_sex = px.bar(df_demo.groupby('sex')['arrest_count'].sum().reset_index(), x="sex", y="arrest_count")
    st.plotly_chart(fig_sex, use_container_width=True)

    st.subheader("Age Group Distribution")
    fig_age = px.bar(df_demo.groupby('age_group')['arrest_count'].sum().reset_index(), x="age_group", y="arrest_count")
    st.plotly_chart(fig_age, use_container_width=True)

    st.subheader("Race Distribution")
    fig_race = px.bar(df_demo.groupby('race')['arrest_count'].sum().reset_index(), x="race", y="arrest_count")
    st.plotly_chart(fig_race, use_container_width=True)

# Borough Comparison Tab
with tabs[2]:
    st.subheader("Borough × Gender")
    fig1 = px.bar(df_boro_sex, x="ARREST_BORO", y="arrest_count", color="PERP_SEX", barmode="group")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Borough × Age")
    fig2 = px.bar(df_boro_age, x="ARREST_BORO", y="arrest_count", color="AGE_GROUP", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Borough × Race")
    fig3 = px.bar(df_boro_race, x="ARREST_BORO", y="arrest_count", color="PERP_RACE", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

# Map Tab
with tabs[3]:
    # Filters moved from sidebar to tab content
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Options: full name + ALL
        boro_options = ["ALL"] + [boro_map[b] for b in sorted(df_map['ARREST_BORO'].dropna().unique())]
        selected_boro_name = st.selectbox("Select Borough", options=boro_options)

    with col2:
        sex_options = ["ALL"] + [sex_map[s] for s in sorted(df_map['PERP_SEX'].dropna().unique())]
        selected_sex_name = st.selectbox("Select Gender", options=sex_options)

    with col3:
        # Date filter (Streamlit slider)
        date_range = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )

    # Apply date filter
    df_map = df_map[
        (df_map['ARREST_DATE'] >= date_range[0]) &
        (df_map['ARREST_DATE'] <= date_range[1])
        ]

    # Apply filters to map data
    map_data = df_map.copy()
    map_data['ARREST_DATE'] = pd.to_datetime(map_data['ARREST_DATE'])
    map_data = map_data[(map_data['ARREST_DATE'] >= date_range[0]) & (map_data['ARREST_DATE'] <= date_range[1])]

    # If a specific Borough is selected, find its abbreviation
    if selected_boro_name != "ALL":
        reverse_boro_map = {v: k for k, v in boro_map.items()}
        selected_boro = reverse_boro_map[selected_boro_name]
        map_data = map_data[map_data['ARREST_BORO'] == selected_boro]

    # If a specific Gender is selected, find its abbreviation
    if selected_sex_name != "ALL":
        reverse_sex_map = {v: k for k, v in sex_map.items()}
        selected_sex = reverse_sex_map[selected_sex_name]
        map_data = map_data[map_data['PERP_SEX'] == selected_sex]

    st.subheader("Borough Arrest Distribution (Choropleth)")
    # Borough Map
    geo_boro_map = {
        "K": "Brooklyn",
        "B": "Bronx",
        "M": "Manhattan",
        "Q": "Queens",
        "S": "Staten Island"
    }
    # Filter data
    df_choro = df_map[df_map['ARREST_BORO'].isin(geo_boro_map.keys())].copy()
    df_choro['ARREST_BORO'] = df_choro['ARREST_BORO'].map(geo_boro_map)
    df_borough_total = df_choro.groupby("ARREST_BORO").size().reset_index(name="arrest_count")

    # Load geojson data
    import requests
    boroughs_geojson = requests.get("https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson").json()

    # Color display, width and height settings
    fig_choro = px.choropleth_mapbox(
        df_borough_total,
        geojson=boroughs_geojson,
        locations="ARREST_BORO",
        featureidkey="properties.BoroName",
        color="arrest_count",
        color_continuous_scale="OrRd",
        mapbox_style="carto-positron",
        center={"lat": 40.7128, "lon": -74.0060},
        zoom=9,
        opacity=0.6
    )
    # Layout
    fig_choro.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600
    )
    # Plot
    st.plotly_chart(fig_choro, use_container_width=True)

    # Hexagon Heatmap
    map_data_h = map_data[['Longitude', 'Latitude']]
    st.subheader("Hexagon Heatmap")
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=40.7128,
            longitude=-74.0060,
            zoom=10,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=map_data_h,
                get_position='[Longitude, Latitude]',
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ],
    ), use_container_width=True)

    # Arrest Points Clustering
    st.subheader("Arrest Points Clustering")
    fig_map = px.scatter_mapbox(
        map_data.sample(min(1000, len(map_data))),
        lat="Latitude", lon="Longitude",
        hover_name="OFNS_DESC", color="ARREST_BORO", zoom=10
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        height=600
    )
    st.plotly_chart(fig_map, use_container_width=True)

# Top Offenses Tab
with tabs[4]:
    st.subheader("Top Offenses by Arrest Count")
    fig_top = px.bar(df_top.sort_values(by="arrest_count", ascending=False),
                     x="offense", y="arrest_count")
    st.plotly_chart(fig_top, use_container_width=True)

    # offense × borough aggregation
    top_offenses = df_map["OFNS_DESC"].value_counts().head(10).index
    df_heat = df_map[df_map["OFNS_DESC"].isin(top_offenses)]

    heat_df = df_heat.groupby(["OFNS_DESC", "ARREST_BORO"]).size().reset_index(name="count")
    heat_pivot = heat_df.pivot(index="OFNS_DESC", columns="ARREST_BORO", values="count").fillna(0)

    # Improved Borough name mapping
    boro_name_map = {
        "B": "Bronx",
        "K": "Brooklyn",
        "M": "Manhattan",
        "Q": "Queens",
        "S": "Staten Island"
    }
    heat_pivot.columns = [boro_name_map.get(b, b) for b in heat_pivot.columns]

    # Heatmap
    fig_heat = px.imshow(
        heat_pivot,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Borough", y="Offense", color="Arrests"),
        title="Top Offenses Across Boroughs"
    )
    # Heatmap layout
    fig_heat.update_layout(
        height=600,
        margin={"r": 0, "t": 50, "l": 50, "b": 0},
        xaxis=dict(tickangle=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# Predictions Tab
with tabs[5]:
    st.subheader("Arrest Forecast")
    
    # Load the pre-trained model
    try:
        with open('arrest_forecast_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # UI controls for forecast
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=360, value=30)
        
        # Make prediction
        forecast = model.predict(model.make_future_dataframe(periods=forecast_days))
        
        # Get historical data
        # This is the key fix - We need to merge historical data from the model
        historical_data = model.history
        
        # Get the date where historical data ends and forecast begins
        last_historical_date = historical_data['ds'].max()

        # Plot forecast
        fig = go.Figure()

        # Historical data points
        fig.add_trace(go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='markers',
            name='Historical Data Points',
            marker=dict(color='blue', size=4)
        ))

        # Historical trend line (red)
        fig.add_trace(go.Scatter(
            x=forecast['ds'][forecast['ds'] <= last_historical_date],
            y=forecast['yhat'][forecast['ds'] <= last_historical_date],
            mode='lines',
            name='Historical Trend',
            line=dict(color='red')
        ))

        # Future prediction line (purple)
        fig.add_trace(go.Scatter(
            x=forecast['ds'][forecast['ds'] > last_historical_date],
            y=forecast['yhat'][forecast['ds'] > last_historical_date],
            mode='lines',
            name='Forecast',
            line=dict(color='purple')
        ))

        # Uncertainty intervals
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'][forecast['ds'] > last_historical_date], 
                         forecast['ds'][forecast['ds'] > last_historical_date].iloc[::-1]]),
            y=pd.concat([forecast['yhat_upper'][forecast['ds'] > last_historical_date], 
                         forecast['yhat_lower'][forecast['ds'] > last_historical_date].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(128,0,128,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Uncertainty Interval'
        ))

        fig.update_layout(
            title='Arrest Volume Forecast',
            xaxis_title='Date',
            yaxis_title='Number of Arrests',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast components
        st.subheader("Forecast Components")
            
        # Create custom component plots instead of using model.plot_components()
        # Weekly component
        weekly_fig = go.Figure()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Extract weekly components
        weekly_effect = forecast['weekly'].tail(7).values
        
        weekly_fig.add_trace(go.Bar(
            x=days,
            y=weekly_effect,
            marker_color='lightblue'
        ))
        weekly_fig.update_layout(
            title='Weekly Pattern',
            xaxis_title='Day of Week',
            yaxis_title='Effect on Arrests'
        )
        
        # Yearly component
        yearly_fig = go.Figure()
        
        # Sort yearly components by date within a year to show Jan-Dec
        yearly_df = forecast[['ds', 'yearly']].copy()
        yearly_df['month'] = yearly_df['ds'].dt.month
        yearly_df['day'] = yearly_df['ds'].dt.day
        yearly_df = yearly_df.sort_values(['month', 'day']).drop_duplicates(['month', 'day'])
        yearly_df = yearly_df.iloc[:365]  # Just use one year's worth
        
        # Format x-axis labels to show month names
        yearly_df['date_label'] = yearly_df['ds'].dt.strftime('%b')
        
        yearly_fig.add_trace(go.Scatter(
            x=yearly_df['ds'],
            y=yearly_df['yearly'],
            mode='lines',
            line=dict(color='green', width=2)
        ))
        yearly_fig.update_layout(
            title='Yearly Pattern',
            xaxis_title='Month',
            yaxis_title='Effect on Arrests',
            xaxis=dict(
                tickformat='%b',
                tickmode='array',
                tickvals=pd.to_datetime([f'2023-{m:02d}-01' for m in range(1, 13)])
            )
        )
        
        # Display the custom component plots
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(weekly_fig, use_container_width=True)
        with col2:
            st.plotly_chart(yearly_fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Insights")
        st.write("The model identifies these seasonal patterns in arrest data:")
        
        # More accurate weekly pattern analysis
        weekday_values = forecast['weekly'][forecast['ds'].dt.dayofweek < 5].mean()  # Mon-Fri (0-4)
        weekend_values = forecast['weekly'][forecast['ds'].dt.dayofweek >= 5].mean()  # Sat-Sun (5-6)
        peak_days = "weekends" if weekend_values > weekday_values else "weekdays"

        st.write("1. **Weekly Pattern**: Arrests tend to peak on " + peak_days)
        
        st.write("2. **Yearly Pattern**: Highest arrest volumes typically occur in " + 
                 ("summer months" if forecast['yearly'].iloc[180:270].mean() > forecast['yearly'].mean() else "winter months"))
                
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first by running 'train_model.py'")

# Crime Predictor Tab
with tabs[6]:
    st.subheader("NYC Violent Crime Prediction Map")
    
    try:
        # Load the violent crime model
        with open('violent_crime_model.pkl', 'rb') as f:
            violent_model_package = pickle.load(f)
            
        # Load neighborhood boundaries from local file
        import json
        try:
            with open('2020_neighborhood_tabulation_areas.geojson', 'r') as f:
                neighborhoods_geojson = json.load(f)
            print("Successfully loaded local neighborhood geojson file")
        except Exception as e:
            st.error(f"Could not load local neighborhood boundaries: {e}")
            
        # Time controls (without requiring button press)
        col1, col2 = st.columns(2)
        
        with col1:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            selected_day_name = st.selectbox("Day of Week", options=days, key="pred_day")
            selected_day = days.index(selected_day_name)
        
        with col2:
            months = ["January", "February", "March", "April", "May", "June", 
                     "July", "August", "September", "October", "November", "December"]
            selected_month_name = st.selectbox("Month", options=months, key="pred_month")
            selected_month = months.index(selected_month_name) + 1
            
        # Get list of neighborhoods from the model features
        neighborhood_names = [f.split('_')[1] for f in violent_model_package['features'] 
                            if f.startswith('NEIGHBORHOOD_')]
        
        # Map tab with loading spinner
        with st.spinner("Generating violent crime predictions for all NYC neighborhoods..."):
            # Pre-compute predictions for all neighborhoods
            neighborhood_predictions = {}
            
            # Create a sample DataFrame with required features
            sample_data = pd.DataFrame({
                'ARREST_DAY': [selected_day] * len(neighborhood_names),
                'ARREST_MONTH': [selected_month] * len(neighborhood_names),
                'ARREST_BORO': ['M'] * len(neighborhood_names),  # Default to Manhattan
                'PERP_SEX': ['M'] * len(neighborhood_names),  # Default to Male
                'PERP_RACE': ['BLACK'] * len(neighborhood_names),  # Default to Black
                'AGE_GROUP': ['25-44'] * len(neighborhood_names)  # Default to 25-44 age group
            })
            
            # Add neighborhood features
            for neighborhood in neighborhood_names:
                sample_data[f'NEIGHBORHOOD_{neighborhood}'] = [1 if n == neighborhood else 0 
                                                             for n in neighborhood_names]
            
            # Get predictions
            probabilities = violent_model_package['pipeline'].predict_proba(sample_data)[:, 1]
            
            # Prepare data for choropleth
            map_data = []
            for i, neighborhood in enumerate(neighborhood_names):
                map_data.append({
                    'neighborhood': neighborhood,
                    'violent_probability': probabilities[i]
                })
                
            map_df = pd.DataFrame(map_data)
            
            # Create visualization tabs
            pred_view_tabs = st.tabs(["Violent Crime Probability Map", "Violent Crime Distribution"])
            
            with pred_view_tabs[0]:
                st.subheader(f"Probability of Violent Crime by Neighborhood ({selected_day_name}, {selected_month_name})")
                
                # Create custom hover text
                map_df['hover_text'] = map_df.apply(
                    lambda x: f"<b>{x['neighborhood']}</b><br>" + 
                             f"Probability of violent crime: {x['violent_probability']:.1%}", 
                    axis=1
                )
                
                # Create choropleth map
                fig = px.choropleth_mapbox(
                    map_df,
                    geojson=neighborhoods_geojson,
                    locations='neighborhood',
                    featureidkey="properties.ntaname",
                    color='violent_probability',
                    color_continuous_scale="OrRd",
                    mapbox_style="carto-positron",
                    zoom=9.5,
                    center={"lat": 40.7128, "lon": -74.0060},
                    opacity=0.7,
                    hover_data={'neighborhood': False, 'violent_probability': False},
                    custom_data=['hover_text']
                )
                
                fig.update_traces(
                    hovertemplate="%{customdata[0]}<extra></extra>"
                )
                
                fig.update_layout(
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                    height=600,
                    coloraxis_colorbar=dict(
                        title="Probability",
                        tickformat=".0%"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with pred_view_tabs[1]:
                st.subheader("Violent Crime Distribution by Neighborhood")
                
                # Sort neighborhoods by probability
                sorted_df = map_df.sort_values('violent_probability', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    sorted_df,
                    x='neighborhood',
                    y='violent_probability',
                    color='violent_probability',
                    color_continuous_scale="OrRd",
                    labels={'violent_probability': 'Probability of Violent Crime'},
                    height=600
                )
                
                fig.update_layout(
                    xaxis_title="Neighborhood",
                    yaxis_title="Probability",
                    yaxis_tickformat=".0%",
                    xaxis_tickangle=45,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first by running 'train_crime_classifier.py'")
