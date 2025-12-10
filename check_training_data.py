import pandas as pd

def analyze_training_data():
    weather_file = "open-meteo-35.19N33.50E87m.csv" # Training weather file
    # Skip rows based on solar_prediction.py logic (skiprows=3)
    try:
        df = pd.read_csv(weather_file, skiprows=3)
    except Exception as e:
        print(f"Error reading {weather_file}: {e}")
        return

    df.columns = [col.strip() for col in df.columns]
    
    # Check cloud cover distribution
    if 'cloud_cover (%)' in df.columns:
        print("Training Data Cloud Cover Statistics:")
        print(df['cloud_cover (%)'].describe())
        
        print("\nHigh Cloud Days (>80%):")
        high_cloud = df[df['cloud_cover (%)'] > 80]
        print(f"Count: {len(high_cloud)}")
        if not high_cloud.empty:
            print(high_cloud[['time', 'cloud_cover (%)', 'shortwave_radiation (W/m²)']].head())
    else:
        print("Cloud cover column not found.")

if __name__ == "__main__":
    analyze_training_data()
