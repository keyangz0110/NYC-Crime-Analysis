import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def train_crime_type_classifier(df):
    print("Training crime type classifier...")
    
    # Print available columns to diagnose the dataset
    print(f"Available columns: {df.columns.tolist()}")
    print(f"First few rows of data:\n{df.head()}")
    
    # Ensure ARREST_DATE is in datetime format
    if 'ARREST_DATE' in df.columns:
        print("Converting ARREST_DATE to datetime...")
        df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'])
        
        # Extract time features from the date
        print("Extracting day of week and month features...")
        df['ARREST_DAY'] = df['ARREST_DATE'].dt.dayofweek
        df['ARREST_MONTH'] = df['ARREST_DATE'].dt.month
    else:
        print("Error: ARREST_DATE column not found in the dataset")
        raise ValueError("ARREST_DATE column not found")
    
    # Determine which column contains crime type
    potential_crime_cols = ['OFNS_DESC', 'PD_DESC', 'LAW_CAT_CD']
    crime_type_col = None
    
    for col in potential_crime_cols:
        if col in df.columns:
            crime_type_col = col
            break
    
    if not crime_type_col:
        print("Error: Could not find crime type column. Available columns:")
        print(df.columns.tolist())
        raise ValueError("No suitable column found for crime type classification")
    
    print(f"Using {crime_type_col} as the target variable for crime type.")
    
    # Define features to use - using only neighborhood, day of week, and month (no hour)
    features = ['NEIGHBORHOOD', 'ARREST_DAY', 'ARREST_MONTH']
    target = crime_type_col
    
    # Filter data to include only rows with valid neighborhoods
    model_data = df.dropna(subset=['NEIGHBORHOOD'] + [target])
    
    # Apply some basic cleaning/grouping to the target to reduce the number of categories
    n_categories = model_data[target].nunique()
    print(f"Original number of crime categories: {n_categories}")
    
    if n_categories > 50:
        # Get the top 30 most common categories and group the rest as "Other"
        top_categories = model_data[target].value_counts().nlargest(30).index
        model_data[target] = model_data[target].apply(lambda x: x if x in top_categories else 'Other')
        print(f"Reduced to {model_data[target].nunique()} categories")
    
    # Encode categorical features
    encoders = {}
    
    X_processed = model_data[features].copy()
    for col in ['NEIGHBORHOOD']:
        encoders[col] = LabelEncoder()
        X_processed[col] = encoders[col].fit_transform(model_data[col])
    
    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(model_data[target])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.25, random_state=42
    )
    
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    print(f"Using features: {features}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("Feature importance:")
    print(feature_importance)
    
    # Save everything needed for prediction
    crime_model_package = {
        'model': model,
        'feature_encoders': encoders,
        'target_encoder': target_encoder,
        'features': features,
        'target_column': target
    }
    
    with open('crime_type_model.pkl', 'wb') as f:
        pickle.dump(crime_model_package, f)
    
    print("Crime type classifier saved as 'crime_type_model.pkl'")
    
    return crime_model_package

def predict_crime_types(neighborhood, day, month, model_package, top_n=3):
    """Predict most likely crime types for given location and time"""
    model = model_package['model']
    feature_encoders = model_package['feature_encoders']
    target_encoder = model_package['target_encoder']
    features = model_package['features']
    
    # Encode neighborhood
    encoded_neighborhood = feature_encoders['NEIGHBORHOOD'].transform([neighborhood])[0]
    
    # Create input data - excluding hour since we don't have it
    input_data = pd.DataFrame({
        'NEIGHBORHOOD': [encoded_neighborhood],
        'ARREST_DAY': [day],
        'ARREST_MONTH': [month]
    })
    
    # Get probabilities for all classes
    probs = model.predict_proba(input_data)[0]
    
    # Get top N classes
    top_indices = np.argsort(probs)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        crime_type = target_encoder.inverse_transform([idx])[0]
        probability = probs[idx]
        results.append((crime_type, probability))
    
    return results
