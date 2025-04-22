# NYC-Crime-Analysis

## Team Members

- Wenhan Jia (wj2255)
- Keyang Zhang (kz2647)
- Haolin Yang (hy2898)

## Abstract

New York City has one of the largest and most active police forces in the world.
Understanding crime trends and arrest patterns is crucial for law enforcement agencies,
policymakers, and citizens to make informed decisions. This project aims to analyze the
New York Police Department (NYPD) arrest dataset to identify crime trends based on
location, time, and type of crime. This study uses big data analytics and visualization to
identify spatial-temporal crime patterns to enhance law enforcement resource allocation
and public safety strategies.

## Problem Statement

Crime distribution across New York City varies significantly based on time and location.
However, it is challenging to identify high-risk areas and peak crime periods without proper analysis.
This project seeks to analyze NYPD arrest data to:

1. Identify spatial patterns in crime distribution across different boroughs.
2. Analyze crime trends over time (hourly, daily, monthly, seasonal).
3. Develop predictive models to estimate high-crime periods and locations using historical data.
4. Visualize results effectively to aid law enforcement planning and public awareness.

## Objectives

### Data Cleaning and Preprocessing

- Handle missing values, outliers, and duplicate records to ensure data integrity.
- Parse time-related fields into a suitable format for temporal analysis and process geographic coordinates for spatial analysis.
- Use Apache Spark for large-scale data processing and Pandas for basic data cleaning.

### Geospatial and Temporal Crime Analysis

- Identify crime-prone areas using KMeans clustering and GIS-based spatial analysis.
- Analyze peak crime hours, daily variations, and seasonal trends using statistical and deep learning techniques (LSTM).

### Temporal Trend Analysis

- Examine crime patterns over time to identify peak hours, days, and seasonal trends.
- Use Matplotlib and Seaborn for trend visualization. Implement LSTM models for time-series forecasting of crime rates.

### Predictive Modeling and Visualization

- Develop machine learning models to forecast high-risk periods and locations.
- Implement an interactive web platform for real-time visualization of crime trends and hotspot mapping.

## Data Source Name and Link, Data File Size, and Approximate Number of Records

- Source Name: NYC OpenData
- Link: [NYPD Arrest Data (Year to Date)](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrest-Data-Year-to-Date-/uip8-fykc/about_data)
- Size: 50MB
- Number of Records: 261000

## Proposed Technologies and Programming Language

### Programming Language

- Python: Main programming language for data processing, analysis, machine learning, and visualization.
- SQL: For querying and managing crime data.
- JavaScript: For web visualization.

### Data Processing and Storage

- Apache Spark: For large-scale data processing and trend analysis.
- SQL: For efficient spatial data storage and retrieval.

### Geospatial Analysis and Mapping

- Kepler.gl: For interactive map-based crime visualization.
- LeafletJS: For mapping and clustering crime locations.

### Machine Learning and Statistical Analysis

- scikit-learn: For clustering (K-Means) and predictive modeling.
- LSTM (PyTorch): For time-series forecasting of crime trends.

### Data Visualization and Dashboarding

- Tableau / Power BI: For interactive data visualization.
- Matplotlib / Seaborn: For statistical crime trend plots.
