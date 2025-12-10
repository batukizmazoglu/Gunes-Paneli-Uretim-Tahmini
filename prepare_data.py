import pandas as pd
import numpy as np
from datetime import datetime

# File paths
solar_file = r'c:\Projeler\panel_uretim_tahmini\Energy and power - PV - Week - Mustafa Kizmazoglu  - 2025-11-29 - 2025-12-05.csv'
weather_file = r'c:\Projeler\panel_uretim_tahmini\open-meteo-35.19N33.50E87m.csv'
output_file = r'c:\Projeler\panel_uretim_tahmini\dataset_final.csv'

def parse_solar_date(date_str):
    # Format: 11/29/2025 12.15 AM
    # Need to replace dot with colon for standard parsing if needed, or use custom format
    # strptime format: %m/%d/%Y %I.%M %p
    return datetime.strptime(date_str, '%m/%d/%Y %I.%M %p')

def clean_and_merge():
    print("Loading Solar Data...")
    # Load solar data
    # Skip first 11 rows to get to header at line 12 (0-indexed 11)
    # The separator is ';'
    try:
        df_solar = pd.read_csv(solar_file, sep=';', skiprows=11, engine='python')
    except Exception as e:
        print(f"Error reading solar file: {e}")
        return

    # Rename columns to be cleaner
    # Expected columns: "Time period", "Power [W]"
    # Sometimes read_csv might pick up extra quotes or spaces, so let's check
    print(f"Solar columns found: {df_solar.columns.tolist()}")
    
    # Clean column names
    df_solar.columns = [c.replace('"', '').strip() for c in df_solar.columns]
    
    # Ensure we have the target columns
    if 'Time period' not in df_solar.columns or 'Power [W]' not in df_solar.columns:
        print("Required columns not found in solar data.")
        # It might be that the file header is slightly different or quoting is an issue.
        # Let's try to find columns by position if name fails, but let's assume name works first.
        return

    # Parse dates
    df_solar['timestamp'] = df_solar['Time period'].apply(parse_solar_date)
    
    # Clean Power column: remove commas, convert to numeric
    # "1,116" -> 1116
    df_solar['power_w'] = df_solar['Power [W]'].astype(str).str.replace(',', '').str.replace('"', '').astype(float)
    
    # Set index
    df_solar = df_solar.set_index('timestamp')
    df_solar = df_solar[['power_w']]
    
    # Sort index just in case
    df_solar = df_solar.sort_index()
    
    print(f"Solar data loaded. shape: {df_solar.shape}")
    print(df_solar.head())

    print("\nLoading Weather Data...")
    # Load weather data
    # Header is at line 4 (index 3). using skiprows=3 ensures we skip first 3 lines and take the 4th as header.
    df_weather = pd.read_csv(weather_file, sep=',', skiprows=3)
    
    print(f"Weather columns found: {df_weather.columns.tolist()}")
    
    # Parse dates
    # Format: 2025-11-29T00:00 (ISO)
    df_weather['timestamp'] = pd.to_datetime(df_weather['time'])
    
    # Set index
    df_weather = df_weather.set_index('timestamp')
    
    # Select relevant columns (drop "time" as it's index now)
    # Keeping all potentially useful weather features
    feature_cols = [
        'temperature_2m (°C)', 
        'shortwave_radiation (W/m²)', 
        'diffuse_radiation (W/m²)', 
        'direct_normal_irradiance (W/m²)', 
        'cloud_cover (%)',
        'is_day ()'
    ]
    
    # Filter only existing columns
    existing_cols = [c for c in feature_cols if c in df_weather.columns]
    df_weather = df_weather[existing_cols]
    
    # Rename columns to be friendlier
    rename_map = {
        'temperature_2m (°C)': 'temp_c',
        'shortwave_radiation (W/m²)': 'shortwave_rad',
        'diffuse_radiation (W/m²)': 'diffuse_rad',
        'direct_normal_irradiance (W/m²)': 'direct_rad',
        'cloud_cover (%)': 'cloud_cover',
        'is_day ()': 'is_day'
    }
    df_weather = df_weather.rename(columns=rename_map)
    
    print(f"Weather data loaded. shape: {df_weather.shape}")
    print(df_weather.head())

    print("\nMerging Data...")
    # Merge on index (timestamp)
    # We want to keep the intersection of times mostly, but since user wants specific week, 
    # let's merge with how indices align. outer join then filter or inner join.
    # Given we prepared both to have proper datetime indices, join should work.
    
    df_final = df_solar.join(df_weather, how='outer')
    
    # Filter for the specific week provided in filename: 2025-11-29 to 2025-12-05
    # The solar file might contain entries exactly at 00:00:00 of the next day or previous?
    # Let's inspect the range.
    print(f"Combined date range: {df_final.index.min()} to {df_final.index.max()}")
    
    # Trimming strict range if needed, or keeping all available data from the solar file timeframe
    # The user request mentioned "using the file... correlated with weather in the same week".
    # So we should probably align to the solar data's extent or the explicit week.
    # Solar data seems to cover exactly that week fully.
    
    # Check for missing values
    print(f"\nMissing values before imputation:\n{df_final.isnull().sum()}")
    
    # Impute missing values
    # For weather data, interpolation is usually safe for small gaps.
    # For power data, if there are gaps, 0 might be wrong if it's daytime, but interpolation is often better than 0.
    # However, if solar data is missing, we might not want to make it up if it's the target variable?
    # The user said "hatasız aynı zamanda boşluksuz" (error-free and gapless).
    # Linear interpolation is a standard gap filling method for time series.
    df_final = df_final.interpolate(method='linear')
    
    # Forward fill / Backward fill any remaining edge cases (like start/end)
    df_final = df_final.ffill().bfill()
    
    print(f"\nMissing values after imputation:\n{df_final.isnull().sum()}")
    
    # Rounding numeric columns to reasonable decimals if needed
    df_final = df_final.round(2)

    # Save to CSV
    df_final.to_csv(output_file)
    print(f"\nSaved processed dataset to: {output_file}")
    print(f"Final shape: {df_final.shape}")
    print("First 5 rows:")
    print(df_final.head())
    print("Last 5 rows:")
    print(df_final.tail())

if __name__ == "__main__":
    clean_and_merge()
