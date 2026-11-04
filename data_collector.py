import fastf1
import pandas as pd
import os

# Configura cache local
os.makedirs('cache_f1', exist_ok=True)
fastf1.Cache.enable_cache('cache_f1')

def collect_season_data(years=[2022, 2023, 2024]):
    all_rows = []
    
    for year in years:
        schedule = fastf1.get_event_schedule(year)
        # Filtra apenas eventos convencionais de corrida
        races = schedule[schedule['EventFormat'] == 'conventional']
        
        for round_num in races['RoundNumber']:
            try:
                session = fastf1.get_session(year, round_num, 'R')
                session.load(laps=False, telemetry=False, weather=True, messages=False)
                
                results = session.results
                air_temp = session.weather_data['AirTemp'].mean()
                track_temp = session.weather_data['TrackTemp'].mean()
                rainfall = session.weather_data['Rainfall'].any()
                
                for _, driver in results.iterrows():
                    all_rows.append({
                        'season': year,
                        'round': round_num,
                        'circuit': session.event['EventName'],
                        'driver': driver['Abbreviation'],
                        'team': driver['TeamName'],
                        'grid_position': driver['GridPosition'],
                        'final_position': driver['Position'],
                        'points': driver['Points'],
                        'status': driver['Status'],
                        'air_temp': air_temp,
                        'track_temp': track_temp,
                        'rain': int(rainfall)
                    })
            except Exception as e:
                print(f"Erro na etapa {round_num}/{year}: {e}")
                
    df = pd.DataFrame(all_rows)
    df.to_csv('f1_dataset_raw.csv', index=False)
    print("Coleta concluída com sucesso!")

if __name__ == "__main__":
    collect_season_data()