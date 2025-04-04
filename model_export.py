import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the data with feature engineering
    """
    # Load data
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Create time-based features
    df['Year'] = df['datetime'].dt.year
    df['Month'] = df['datetime'].dt.month
    df['Day'] = df['datetime'].dt.day
    df['Hour'] = df['datetime'].dt.hour
    
    # Create categorical features
    # Define rush_hour: Morning (7-9), Lunch (11-13), Evening (17-18), else Non-Rush Hour
    df['rush_hour'] = df['Hour'].apply(
        lambda x: 'Morning' if 7 <= x <= 9 
        else ('Lunch' if 11 <= x <= 13 
              else ('Evening' if 17 <= x <= 18 else 'Non-Rush Hour'))
    )
    
    # Define weather_category: Based on temperature
    df['weather_category'] = df['temp'].apply(
        lambda x: 'Hot' if 28 <= x <= 45 
        else ('Cold' if 0 <= x <= 15 else 'Mild')
    )
    
    # Define wind_category: Based on windspeed
    df['wind_category'] = df['windspeed'].apply(
        lambda x: 'Windy' if 25 <= x <= 60 else 'Mild Windy'
    )
    
    # Convert selected columns to categorical dtype
    for col in ["season", "weather", "rush_hour", "weather_category", "wind_category"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    return df

def prepare_features(df, is_train=True, feature_names=None):
    """
    Prepare features for model training or prediction
    """
    # Define columns to drop
    if is_train:
        drop_cols = ["datetime", "count", "casual", "registered"]
    else:
        drop_cols = ["datetime"]
    
    # Drop unnecessary columns
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # If feature_names is provided, ensure consistent features
    if feature_names is not None:
        # Add missing columns with zeros
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Keep only the columns that were in the training data
        X = X[feature_names]
    
    # Return features and target if training
    if is_train:
        y = df["count"]
        return X, y
    else:
        return X

def train_and_save_model(train_file, model_dir="model"):
    """
    Train the model and save it along with feature names
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess training data
    train_data = load_and_preprocess_data(train_file)
    
    # Prepare features
    X, y = prepare_features(train_data, is_train=True)
    
    # Save feature names for consistent inference
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))
    
    # Train the model
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, os.path.join(model_dir, "random_forest_model.joblib"))
    
    print(f"Model and feature names saved to {model_dir}/")
    
    return model, feature_names

def prepare_single_sample(datetime, season, holiday, workingday, weather, temp, 
                          atemp, humidity, windspeed, feature_names):
    """
    Prepare a single sample for prediction
    """
    # Create a DataFrame with a single row
    df = pd.DataFrame({
        'datetime': [pd.Timestamp(datetime)],
        'season': [season],
        'holiday': [holiday],
        'workingday': [workingday],
        'weather': [weather],
        'temp': [temp],
        'atemp': [atemp],
        'humidity': [humidity],
        'windspeed': [windspeed]
    })
    
    # Apply the same preprocessing steps directly without using load_and_preprocess_data
    # Sort and reset index (not needed for a single row but kept for consistency)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Create time-based features
    df['Year'] = df['datetime'].dt.year
    df['Month'] = df['datetime'].dt.month
    df['Day'] = df['datetime'].dt.day
    df['Hour'] = df['datetime'].dt.hour
    
    # Create categorical features
    # Rush hour
    df['rush_hour'] = df['Hour'].apply(
        lambda x: 'Morning' if 7 <= x <= 9 
        else ('Lunch' if 11 <= x <= 13 
              else ('Evening' if 17 <= x <= 18 else 'Non-Rush Hour'))
    )
    
    # Weather category
    df['weather_category'] = df['temp'].apply(
        lambda x: 'Hot' if 28 <= x <= 45 
        else ('Cold' if 0 <= x <= 15 else 'Mild')
    )
    
    # Wind category
    df['wind_category'] = df['windspeed'].apply(
        lambda x: 'Windy' if 25 <= x <= 60 else 'Mild Windy'
    )
    
    # Convert selected columns to categorical
    for col in ["season", "weather", "rush_hour", "weather_category", "wind_category"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    # Prepare features
    X = prepare_features(df, is_train=False, feature_names=feature_names)
    
    return X

def load_model(model_dir="model"):
    """
    Load the saved model and feature names
    """
    model = joblib.load(os.path.join(model_dir, "random_forest_model.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    
    return model, feature_names

if __name__ == "__main__":
    # This is used to train and save the model
    train_and_save_model("train.csv")