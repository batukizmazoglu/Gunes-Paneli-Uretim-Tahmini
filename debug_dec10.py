import json
import pandas as pd
import numpy as np

def analyze_dec10():
    with open('forecast_data.json', 'r') as f:
        data = json.load(f)
    
    minutely = data['minutely_15']
    df = pd.DataFrame(minutely)
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter for Dec 10
    mask = (df['time'] >= '2025-12-10') & (df['time'] < '2025-12-11')
    df_day = df.loc[mask].copy()
    
    if df_day.empty:
        print("No data for Dec 10")
        return

    # Print hourly averages to see "what the model saw"
    df_day['hour'] = df_day['time'].dt.hour
    
    print(f"{'Hour':<5} | {'Shortwave (W/m2)':<20} | {'Direct (W/m2)':<15} | {'Diffuse (W/m2)':<15} | {'Cloud (%)':<10} | {'Temp (C)':<10}")
    print("-" * 90)
    
    hourly = df_day.groupby('hour').mean()
    
    for h in range(24):
        if h in hourly.index:
            row = hourly.loc[h]
            print(f"{h:<5} | {row['shortwave_radiation']:>20.1f} | {row['direct_normal_irradiance']:>15.1f} | {row['diffuse_radiation']:>15.1f} | {row['cloud_cover']:>10.0f} | {row['temperature_2m']:>10.1f}")

if __name__ == "__main__":
    analyze_dec10()
