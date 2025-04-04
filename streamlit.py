import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time, datetime
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# Set page configuration
st.set_page_config(
    page_title="Bike Rental Prediction App",
    page_icon="🚲",
    layout="wide"
)

# ====== MODEL FUNCTIONS - Included directly to avoid import issues ======

def load_and_preprocess_data(data_source):
    """
    Load and preprocess the data with feature engineering
    
    Parameters:
    data_source (str or pd.DataFrame): Either a file path to a CSV or a pandas DataFrame
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame with engineered features
    """
    # Load data from file or use provided DataFrame
    if isinstance(data_source, str):
        # It's a file path
        df = pd.read_csv(data_source)
    else:
        # It's already a DataFrame
        df = data_source.copy()
    
    # Convert datetime and sort
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
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Define columns to drop
    if is_train:
        drop_cols = ["datetime", "count", "casual", "registered"]
    else:
        drop_cols = ["datetime"]
    
    # Drop unnecessary columns
    X = df_copy.drop(columns=[col for col in drop_cols if col in df_copy.columns])
    
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
        y = df_copy["count"]
        return X, y
    else:
        return X

def prepare_single_sample(datetime_val, season, holiday, workingday, weather, temp, 
                          atemp, humidity, windspeed, feature_names):
    """
    Prepare a single sample for prediction
    """
    # Create a DataFrame with a single row
    sample_data = {
        'datetime': [pd.Timestamp(datetime_val)],
        'season': [season],
        'holiday': [holiday],
        'workingday': [workingday],
        'weather': [weather],
        'temp': [temp],
        'atemp': [atemp],
        'humidity': [humidity],
        'windspeed': [windspeed]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Process the DataFrame directly
    processed_df = load_and_preprocess_data(df)
    
    # Prepare features - make sure we pass the DataFrame, not a method
    X = prepare_features(processed_df, is_train=False, feature_names=feature_names)
    
    return X

def load_model(model_dir="model"):
    """
    Load the saved model and feature names
    """
    try:
        model = joblib.load(os.path.join(model_dir, "random_forest_model.joblib"))
        feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
        return model, feature_names
    except FileNotFoundError:
        st.error(f"Model files not found in directory: {model_dir}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to convert DataFrame to CSV for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ====== STREAMLIT APP ======

# App title and description
st.title("🚲 Bike Rental Demand Prediction")
st.markdown("""
This app predicts the number of bike rentals based on weather and time features.
You can make individual predictions or upload a CSV file for batch predictions.
""")

# Load model at startup
with st.spinner("Loading model..."):
    model, feature_names = load_model()
    
    if model is not None and feature_names is not None:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Failed to load model. Please check that model files exist in the 'model' directory.")
        st.stop()

# Create tabs for Single Prediction and Batch Prediction
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Make a Single Prediction")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # Datetime input with more control
    with col1:
        prediction_date = st.date_input("Date", datetime.now().date())
        
        # Use more explicit time selection with sliders
        hour = st.slider("Hour", 0, 23, datetime.now().hour)
        minute = st.slider("Minute", 0, 59, datetime.now().minute)
        
        # Create time object and combine with date
        prediction_time = time(hour, minute)
        st.write(f"Selected time: {prediction_time.strftime('%H:%M')}")
        
        # Combine date and time - convert to string first to avoid any object reference issues
        date_str = prediction_date.strftime('%Y-%m-%d')
        time_str = prediction_time.strftime('%H:%M:%S')
        datetime_val = pd.Timestamp(f"{date_str} {time_str}")
        
        # Weather inputs
        season = st.selectbox("Season", [1, 2, 3, 4], 
                             format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x])
        weather = st.selectbox("Weather", [1, 2, 3, 4], 
                              format_func=lambda x: {
                                  1: "Clear", 
                                  2: "Mist/Cloudy", 
                                  3: "Light Rain/Snow", 
                                  4: "Heavy Rain/Snow/Fog"
                              }[x])
    
    with col2:
        holiday = st.checkbox("Holiday")
        workingday = st.checkbox("Working Day")
        
        # Environmental factors
        temp = st.slider("Temperature (°C)", 0.0, 45.0, 20.0)
        atemp = st.slider("Feels Like Temperature (°C)", 0.0, 45.0, 22.0)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        windspeed = st.slider("Windspeed (km/h)", 0.0, 60.0, 10.0)
    
    # Prediction button
    if st.button("Predict Bike Rentals"):
        with st.spinner("Calculating prediction..."):
            # Prepare the sample for prediction
            X = prepare_single_sample(
                datetime_val, 
                season, 
                int(holiday), 
                int(workingday), 
                weather, 
                temp, 
                atemp, 
                humidity, 
                windspeed, 
                feature_names
            )
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Ensure no negative predictions
            prediction = max(0, prediction)
            
            # Display prediction
            st.success(f"Predicted number of bike rentals: **{int(prediction)}**")
            
            # Display time-based context
            hour_val = datetime_val.hour
            rush_hour = 'Morning' if 7 <= hour_val <= 9 else ('Lunch' if 11 <= hour_val <= 13 else ('Evening' if 17 <= hour_val <= 18 else 'Non-Rush Hour'))
            st.info(f"Time Context: {rush_hour} rush hour")
            
            # Display weather context
            weather_category = 'Hot' if 28 <= temp <= 45 else ('Cold' if 0 <= temp <= 15 else 'Mild')
            wind_category = 'Windy' if 25 <= windspeed <= 60 else 'Mild Windy'
            st.info(f"Weather Context: {weather_category} temperature, {wind_category}")

with tab2:
    st.header("Make Batch Predictions")
    
    # File upload section with clear instructions
    st.write("Upload a CSV file for batch predictions or use the demo data.")
    
    # Create demo data
    if st.button("Generate Demo Data"):
        # Create sample data for demonstration
        demo_data = {
            'datetime': [
                '2023-01-01 08:00:00', 
                '2023-01-01 12:00:00',
                '2023-01-01 18:00:00',
                '2023-01-02 08:00:00',
                '2023-01-02 17:00:00'
            ],
            'season': [1, 1, 1, 1, 1],
            'holiday': [0, 0, 0, 0, 0],
            'workingday': [1, 1, 1, 1, 1],
            'weather': [1, 2, 1, 3, 2],
            'temp': [12.5, 15.8, 14.2, 10.5, 13.2],
            'atemp': [14.0, 17.5, 15.8, 12.0, 14.5],
            'humidity': [60, 55, 65, 70, 62],
            'windspeed': [10.5, 12.8, 8.5, 15.2, 11.0]
        }
        
        demo_df = pd.DataFrame(demo_data)
        st.session_state['demo_df'] = demo_df
        st.success("Demo data generated! You can now process it below.")
        
        # Display the demo data
        st.subheader("Preview of Demo Data")
        st.dataframe(demo_df)
        
        if st.button("Process Demo Data"):
            with st.spinner("Processing demo data..."):
                # Make a copy to avoid reference issues
                test_data = demo_df.copy()
                
                # Convert datetime column
                test_data['datetime'] = pd.to_datetime(test_data['datetime'])
                
                # Process the data - pass the DataFrame directly
                preprocessed_data = load_and_preprocess_data(test_data)
                
                # Prepare features - pass the processed DataFrame directly
                X_test = prepare_features(preprocessed_data, is_train=False, feature_names=feature_names)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Ensure no negative predictions
                predictions = np.maximum(predictions, 0).astype(int)
                
                # Create results DataFrame
                results = pd.DataFrame({
                    'datetime': preprocessed_data['datetime'],
                    'count_prediction': predictions
                })
                
                # Display the results
                st.subheader("Prediction Results")
                st.dataframe(results)
                
                # Visualize predictions
                st.subheader("Visualization of Predictions")
                fig, ax = plt.subplots(figsize=(10, 6))
                results.set_index('datetime')['count_prediction'].plot(ax=ax)
                ax.set_xlabel('Date and Time')
                ax.set_ylabel('Predicted Bike Rentals')
                ax.set_title('Predicted Bike Rentals Over Time')
                st.pyplot(fig)
                
                # Provide download link
                csv = convert_df_to_csv(results)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="bike_rental_predictions.csv",
                    mime="text/csv"
                )
    else:
        # Upload file option
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                test_data = pd.read_csv(uploaded_file)
                
                # Display the first few rows of the uploaded file
                st.subheader("Preview of Uploaded Data")
                st.dataframe(test_data.head())
                
                # Check if the required columns are present
                required_columns = ["datetime", "season", "holiday", "workingday", 
                                  "weather", "temp", "atemp", "humidity", "windspeed"]
                
                missing_columns = [col for col in required_columns if col not in test_data.columns]
                
                if missing_columns:
                    st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
                else:
                    if st.button("Process Uploaded Data"):
                        with st.spinner("Processing uploaded data..."):
                            # Make a copy to avoid reference issues
                            df_copy = test_data.copy()
                            
                            # Convert datetime column
                            if not pd.api.types.is_datetime64_dtype(df_copy['datetime']):
                                df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
                            
                            # Process the data - pass the DataFrame directly
                            preprocessed_data = load_and_preprocess_data(df_copy)
                            
                            # Prepare features - pass the processed DataFrame directly
                            X_test = prepare_features(preprocessed_data, is_train=False, feature_names=feature_names)
                            
                            # Make predictions
                            predictions = model.predict(X_test)
                            
                            # Ensure no negative predictions
                            predictions = np.maximum(predictions, 0).astype(int)
                            
                            # Create results DataFrame
                            results = pd.DataFrame({
                                'datetime': preprocessed_data['datetime'],
                                'count_prediction': predictions
                            })
                            
                            # Display the results
                            st.subheader("Prediction Results")
                            st.dataframe(results)
                            
                            # Visualize predictions
                            st.subheader("Visualization of Predictions")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            results.set_index('datetime')['count_prediction'].plot(ax=ax)
                            ax.set_xlabel('Date and Time')
                            ax.set_ylabel('Predicted Bike Rentals')
                            ax.set_title('Predicted Bike Rentals Over Time')
                            st.pyplot(fig)
                            
                            # Provide download link
                            csv = convert_df_to_csv(results)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name="bike_rental_predictions.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please make sure your CSV file has the required columns and format.")

# Add information about the model in the sidebar
st.sidebar.header("About the Model")
st.sidebar.markdown("""
This application uses a Random Forest Regressor trained on historical bike rental data.

**Model Details:**
- Algorithm: Random Forest
- Number of Trees: 50
- Max Depth: 8

**Features Used:**
- Date and time features
- Weather conditions
- Temperature and humidity
- Holiday/working day indicators
""")

# Add CSV format information
with st.sidebar.expander("CSV Format Guidelines"):
    st.markdown("""
    Your CSV should have these columns:
    - datetime (YYYY-MM-DD HH:MM:SS)
    - season (1-4)
    - holiday (0/1)
    - workingday (0/1)
    - weather (1-4)
    - temp (°C)
    - atemp (°C)
    - humidity (%)
    - windspeed (km/h)
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Bike Rental Prediction App © 2025")