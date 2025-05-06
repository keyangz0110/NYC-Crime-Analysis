import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTENC
import pickle
from datetime import datetime
import holidays
from category_encoders import TargetEncoder
import seaborn as sns
import matplotlib.pyplot as plt

def map_to_violent_category(crime_type):
    """Map specific crime types to violent vs non-violent categories"""
    crime_type = str(crime_type).strip()
    
    # Violent crimes
    violent_crimes = [
        'Assault 3 & Related Offenses',
        'Felony Assault',
        'Murder & Non-Negl. Manslaughte',
        'Offenses Against The Person',
        'Rape',
        'Sex Crimes',
        'Dangerous Weapons',
        'Unlawful Poss. Weap. On School'
    ]
    
    # Map to binary category
    if crime_type in violent_crimes:
        return 'Violent'
    else:
        return 'Non-Violent'

def add_historical_features(df, train_df=None, alpha=10):
    """Add historical frequency features using smoothed probabilities"""
    if train_df is None:
        # If no training data provided, use the same data
        train_df = df
    
    # Compute counts per (NEIGHBORHOOD, DAY, MONTH, CATEGORY)
    counts = (train_df
        .groupby(['NEIGHBORHOOD', 'ARREST_DAY', 'ARREST_MONTH', 'VIOLENT_CATEGORY_ENCODED'])
        .size()
        .unstack(fill_value=0))
    
    # Compute global probabilities
    global_probs = train_df['VIOLENT_CATEGORY_ENCODED'].value_counts(normalize=True)
    
    # Compute smoothed probabilities
    prob_df = (counts + alpha * global_probs) \
              .div(counts.sum(axis=1) + alpha, axis=0) \
              .reset_index()
    
    # Merge probabilities back into the dataset
    df_with_probs = df.merge(
        prob_df,
        on=['NEIGHBORHOOD', 'ARREST_DAY', 'ARREST_MONTH'],
        how='left'
    )
    
    # Fill any missing probabilities with global probabilities
    for category in global_probs.index:
        col_name = f'PROB_{category}'
        if col_name in df_with_probs.columns:
            df_with_probs[col_name] = df_with_probs[col_name].fillna(global_probs[category])
    
    # Add neighborhood-specific features without the prefix
    for neighborhood in df['NEIGHBORHOOD'].unique():
        df_with_probs[f'NEIGHBORHOOD_{neighborhood}'] = (df_with_probs['NEIGHBORHOOD'] == neighborhood).astype(int)
    
    return df_with_probs

def find_optimal_threshold(y_true, y_proba, target_recall=0.6):
    """Find the optimal threshold that achieves target recall while maximizing precision"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find all thresholds that achieve at least target recall
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) == 0:
        print(f"Warning: Could not achieve target recall of {target_recall}")
        # Return threshold that gives maximum recall
        max_recall_idx = np.argmax(recall)
        return thresholds[max_recall_idx], f1_score(y_true, (y_proba >= thresholds[max_recall_idx]).astype(int))
    
    # Among valid thresholds, find the one with highest precision
    best_idx = valid_indices[np.argmax(precision[valid_indices])]
    optimal_threshold = thresholds[best_idx]
    
    # Calculate F1 score at this threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    
    # Print optimization details
    print(f"\nThreshold optimization details:")
    print(f"Target recall: {target_recall}")
    print(f"Achieved recall: {recall[best_idx]:.3f}")
    print(f"Precision at this threshold: {precision[best_idx]:.3f}")
    
    return optimal_threshold, f1

def train_violent_crime_classifier(df, use_smote_nc=False, alpha=10, target_recall=0.6):
    """Train a binary classifier for violent vs non-violent crimes"""
    print("Training violent crime classifier...")
    
    try:
        # Print available columns to diagnose the dataset
        print(f"Available columns: {df.columns.tolist()}")
        print(f"First few rows of data:\n{df.head()}")
        
        # Create a copy of the input DataFrame to avoid modifying the original
        df = df.copy()
        
        # Ensure ARREST_DATE is in datetime format
        if 'ARREST_DATE' in df.columns:
            print("Converting ARREST_DATE to datetime...")
            df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'])
            
            # Extract temporal features from date
            print("Extracting temporal features...")
            df['ARREST_DAY'] = df['ARREST_DATE'].dt.dayofweek
            df['ARREST_MONTH'] = df['ARREST_DATE'].dt.month
        else:
            raise ValueError("ARREST_DATE column not found in the dataset")
        
        # Handle demographic information
        print("\nProcessing demographic information...")
        if 'PERP_SEX' in df.columns:
            df['PERP_SEX'] = df['PERP_SEX'].fillna('U')
        else:
            print("Warning: PERP_SEX column not found")
        
        if 'PERP_RACE' in df.columns:
            df['PERP_RACE'] = df['PERP_RACE'].fillna('UNKNOWN')
        else:
            print("Warning: PERP_RACE column not found")
            
        if 'AGE_GROUP' in df.columns:
            df['AGE_GROUP'] = df['AGE_GROUP'].fillna('UNKNOWN')
        else:
            print("Warning: AGE_GROUP column not found")
        
        # Determine which column contains crime type
        potential_crime_cols = ['OFNS_DESC', 'PD_DESC', 'LAW_CAT_CD']
        crime_type_col = None
        
        for col in potential_crime_cols:
            if col in df.columns:
                crime_type_col = col
                break
        
        if not crime_type_col:
            raise ValueError("No suitable column found for crime type classification")
        
        print(f"Using {crime_type_col} as the target variable for crime type.")
        
        # Define feature sets using only available columns
        categorical_features = ['ARREST_DAY', 'ARREST_MONTH', 'ARREST_BORO', 'PERP_SEX', 'PERP_RACE', 'AGE_GROUP']
        
        # Add neighborhood-specific features if available
        if 'NEIGHBORHOOD' in df.columns:
            for neighborhood in df['NEIGHBORHOOD'].unique():
                df[f'NEIGHBORHOOD_{neighborhood}'] = (df['NEIGHBORHOOD'] == neighborhood).astype(int)
                categorical_features.append(f'NEIGHBORHOOD_{neighborhood}')
        
        features = categorical_features
        
        # Filter data and create a new DataFrame for modeling
        model_data = df.dropna(subset=[crime_type_col]).copy()
        
        # Map crimes to violent/non-violent categories
        print("\nMapping crimes to violent/non-violent categories...")
        model_data['VIOLENT_CATEGORY'] = model_data[crime_type_col].apply(map_to_violent_category)
        
        # Initialize label encoder for the target variable
        label_encoder = LabelEncoder()
        model_data['VIOLENT_CATEGORY_ENCODED'] = label_encoder.fit_transform(model_data['VIOLENT_CATEGORY'])
        
        # Split data
        X = model_data[features]
        y = model_data['VIOLENT_CATEGORY_ENCODED']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Reset indexes to ensure they match
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # Undersample majority class in training data
        print("\nBalancing classes through undersampling...")
        violent_indices = y_train[y_train == 1].index
        non_violent_indices = y_train[y_train == 0].index
        
        # Randomly sample non-violent cases to match violent count
        np.random.seed(42)
        sampled_non_violent_indices = np.random.choice(
            non_violent_indices,
            size=len(violent_indices),
            replace=False
        )
        
        # Combine indices
        balanced_indices = np.concatenate([violent_indices, sampled_non_violent_indices])
        
        # Create balanced training set
        X_train_balanced = X_train.loc[balanced_indices]
        y_train_balanced = y_train.loc[balanced_indices]
        
        print(f"Original training set size: {len(X_train)}")
        print(f"Balanced training set size: {len(X_train_balanced)}")
        print(f"Violent cases in balanced set: {sum(y_train_balanced == 1)}")
        print(f"Non-Violent cases in balanced set: {sum(y_train_balanced == 0)}")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
        
        # Calculate class weights
        class_counts = np.bincount(y_train)
        # Base weights for class balancing
        base_weights = {i: sum(class_counts) / (len(class_counts) * count) 
                       for i, count in enumerate(class_counts)}
        
        # Additional weight multiplier for violent class (index 1)
        violent_weight_multiplier = 2.0  # Increase this to make model more conservative about violent predictions
        class_weights = base_weights.copy()
        class_weights[1] *= violent_weight_multiplier  # Increase weight for violent class
        
        print(f"\nClass weights:")
        print(f"Non-Violent (class 0): {class_weights[0]:.2f}")
        print(f"Violent (class 1): {class_weights[1]:.2f} (base weight * {violent_weight_multiplier})")
        
        # Create final pipeline
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', CatBoostClassifier(
                iterations=1000,
                learning_rate=0.01,
                depth=12,
                l2_leaf_reg=1,
                random_state=42,
                task_type='GPU',
                devices='0:1',
                verbose=False,
                class_weights=class_weights
            ))
        ])
        
        # Cross-validation score with stratification
        cv_scores = cross_val_score(
            final_pipeline, 
            X_train,
            y_train,
            cv=5, 
            scoring='f1_weighted',
            error_score='raise'
        )
        print(f"\nCross-validation scores (weighted F1): {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Fit the final pipeline
        print("\nFitting final model...")
        final_pipeline.fit(X_train, y_train)
        
        # Get feature importance
        feature_names = final_pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = final_pipeline.named_steps['classifier'].feature_importances_
        
        # Create DataFrame with feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        print(feature_importance.head(20))
        
        # Get probability predictions
        y_proba = final_pipeline.predict_proba(X_test)[:, 1]  # Probability of violent class
        
        # Find optimal threshold
        optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_proba, target_recall)
        print(f"\nOptimal threshold: {optimal_threshold:.3f}")
        print(f"F1 score at optimal threshold: {optimal_f1:.3f}")
        
        # Evaluate with optimal threshold
        y_pred = (y_proba >= optimal_threshold).astype(int)
        print("\nClassification Report (with optimal threshold):")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Create and display confusion matrix
        print("\nConfusion Matrix (with optimal threshold):")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title('Violent Crime Classifier Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('violent_crime_confusion_matrix.png')
        plt.close()
        
        # Save model package with threshold
        violent_model_package = {
            'pipeline': final_pipeline,
            'label_encoder': label_encoder,
            'features': features,
            'target_column': 'VIOLENT_CATEGORY',
            'categorical_features': categorical_features,
            'feature_importance': feature_importance,
            'use_smote_nc': use_smote_nc,
            'optimal_threshold': optimal_threshold,
            'class_weights': class_weights
        }
        
        with open('violent_crime_model.pkl', 'wb') as f:
            pickle.dump(violent_model_package, f)
        
        print("Violent crime classifier saved as 'violent_crime_model.pkl'")
        
        return violent_model_package
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise
