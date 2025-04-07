import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.interpolate import interp1d

# Enable caching
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

# Get available races and track lengths (approximate, in km)
TRACK_LENGTHS = {
    'Bahrain': 5.412, 'Saudi Arabia': 6.174, 'Australia': 5.278, 'Azerbaijan': 6.003, 'Miami': 5.412,
    'Monaco': 3.337, 'Spain': 4.675, 'Canada': 4.361, 'Austria': 4.318, 'Great Britain': 5.891,
    'Hungary': 4.381, 'Belgium': 7.004, 'Netherlands': 4.259, 'Italy': 5.793, 'Singapore': 5.063,
    'Japan': 5.807, 'Qatar': 5.419, 'United States': 5.513, 'Mexico': 4.304, 'Brazil': 4.309,
    'Las Vegas': 6.201, 'Abu Dhabi': 5.281
}

def get_available_races(year):
    try:
        events = fastf1.get_event_schedule(year)
        return {row['EventName'].replace(" Grand Prix", ""): row['RoundNumber'] 
                for _, row in events.iterrows() if row['Session5'] == 'Race'}
    except Exception as e:
        print(f"Error fetching races: {e}")
        return {}

# Load race session with validation
def load_race_data(year, grand_prix, driver, rival):
    try:
        session = fastf1.get_session(year, grand_prix, 'R')
        session.load(telemetry=True, laps=True, weather=True)
        driver_laps = session.laps.pick_driver(driver)
        rival_laps = session.laps.pick_driver(rival) if rival else None
        
        if driver_laps.empty:
            raise ValueError(f"No lap data for {driver} in {year} {grand_prix}")
        if rival and (rival_laps is None or rival_laps.empty):
            raise ValueError(f"No lap data for {rival} in {year} {grand_prix}")
        
        telemetry = driver_laps.get_telemetry()
        weather = session.weather_data
        track_length = TRACK_LENGTHS.get(grand_prix, 5.0)  # Default 5 km if unknown
        print(f"Loaded {len(driver_laps)} laps for {driver}" + 
              (f" and {len(rival_laps)} for {rival}" if rival else "") +
              f" at {grand_prix} ({track_length} km)")
        return driver_laps, rival_laps, telemetry, weather, track_length
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

# Tire degradation (realistic rates per km, adjusted by weather)
def calculate_tire_wear(laps, weather, track_length, 
                       compound_factor={'SOFT': 0.025, 'MEDIUM': 0.015, 'HARD': 0.010}):  # % wear per km
    if laps is None or weather is None:
        return None
    
    compound = laps['Compound'].mode()[0]
    base_wear_rate = compound_factor.get(compound, 0.015)  # Default to medium
    
    # Interpolate weather to lap timestamps
    weather['TimeSec'] = weather['Time'].dt.total_seconds()
    laps['LapStartSec'] = laps['LapStartTime'].dt.total_seconds()
    weather_interp = interp1d(weather['TimeSec'], weather['Rainfall'], bounds_error=False, fill_value=0)
    rain_factor = weather_interp(laps['LapStartSec'])
    
    # Wear increases with rain (up to 50% more)
    wear_per_lap = base_wear_rate * track_length * (1 + 0.5 * rain_factor)
    laps['TireWear'] = np.cumsum(wear_per_lap)
    laps['TireWear'] = np.clip(laps['TireWear'], 0, 100)
    return laps

# Performance factors (fuel and fatigue)
def calculate_performance_factors(laps, track_length, initial_fuel=105):  # 2023 fuel limit in kg
    laps_per_tank = initial_fuel / (0.75 * track_length)  # Approx 0.75 kg/km fuel burn
    fuel_per_lap = initial_fuel / min(len(laps), laps_per_tank)
    laps['FuelLoad'] = np.linspace(initial_fuel, max(0, initial_fuel - fuel_per_lap * len(laps)), len(laps))
    laps['FuelEffect'] = 1 + 0.035 * (laps['FuelLoad'] / initial_fuel)  # 0.035s per kg (F1 estimate)
    laps['FatigueFactor'] = 1 + np.linspace(0, 0.05, len(laps))  # 5% max fatigue (more realistic)
    return laps

# Race simulation
def simulate_race(laps, weather, track_length, pit_stops, pit_time=22, track_temp=30):
    if laps is None or weather is None:
        return None, None, None
    
    race_laps = laps.copy()
    race_laps['LapTime'] = pd.to_timedelta(race_laps['LapTime'], errors='coerce')
    race_laps['AdjustedLapTime'] = race_laps['LapTime'].fillna(pd.Timedelta(seconds=90))  # Default for NaN
    
    temp_factor = 1 + 0.002 * (track_temp - 30)  # 0.002s per °C (F1 estimate)
    stints = []
    stint_start = 0
    
    for i, lap in race_laps.iterrows():
        rain_effect = 1 + 0.2 * weather['Rainfall'].iloc[min(i, len(weather)-1)]  # Reduced rain impact
        wear_effect = 1 + 0.005 * (lap['TireWear'] / 100)  # 0.5% time loss per 100% wear
        race_laps.loc[i, 'AdjustedLapTime'] *= (lap['FatigueFactor'] * rain_effect * temp_factor *
                                                lap['FuelEffect'] * wear_effect)
    
    for pit_lap in sorted(pit_stops + [len(race_laps) + 1]):
        if pit_lap <= len(race_laps):
            idx = race_laps.index[race_laps['LapNumber'] == pit_lap][0]
            race_laps.loc[idx, 'AdjustedLapTime'] += pd.Timedelta(seconds=pit_time)
            stint = race_laps.iloc[stint_start:idx]
            if not stint.empty:
                stints.append({
                    'laps': stint['LapNumber'].tolist(),
                    'avg_time': stint['AdjustedLapTime'].mean().total_seconds(),
                    'tire_wear_end': stint['TireWear'].iloc[-1]
                })
            post_pit_mask = race_laps['LapNumber'] > pit_lap
            if post_pit_mask.any():
                race_laps.loc[post_pit_mask, 'TireWear'] = np.linspace(0, track_length * 0.015 * post_pit_mask.sum(), 
                                                                      post_pit_mask.sum())
            stint_start = idx + 1
    
    total_time = race_laps['AdjustedLapTime'].sum()
    return race_laps, total_time, stints

# Strategy optimization
def optimize_strategy(laps, weather, track_length, max_pits=2):
    if laps is None:
        return None, None, None
    best_time = None
    best_strategy = []
    results = []
    
    total_laps = len(laps)
    for n_pits in range(1, min(max_pits + 1, total_laps // 10)):  # Limit pits by race length
        pit_options = np.linspace(total_laps * 0.2, total_laps * 0.8, n_pits + 2, dtype=int)[1:-1]
        for pit_comb in [pit_options[i:i+n_pits] for i in range(len(pit_options) - n_pits + 1)]:
            _, total_time, _ = simulate_race(laps, weather, track_length, pit_comb)
            if total_time is not None:
                results.append((list(pit_comb), total_time))
                if best_time is None or total_time < best_time:
                    best_time = total_time
                    best_strategy = list(pit_comb)
    return best_strategy, best_time, results

# Interactive dashboard
def create_dashboard(simulated_laps, rival_laps, pit_stops, stints, driver, rival, grand_prix, year, best_strategy, total_time):
    if simulated_laps is None:
        return
    
    fig = make_subplots(rows=3, cols=2, 
                        subplot_titles=("Lap Times", "Tire Wear", "Track Map", "Stint Breakdown", 
                                        "Weather Impact", "Strategy Comparison"))
    
    # Lap Times
    fig.add_trace(go.Scatter(x=simulated_laps['LapNumber'], 
                            y=simulated_laps['AdjustedLapTime'].dt.total_seconds(),
                            mode='lines+markers', name=driver, line=dict(color='blue')),
                  row=1, col=1)
    if rival_laps is not None:
        fig.add_trace(go.Scatter(x=rival_laps['LapNumber'], 
                                y=rival_laps['AdjustedLapTime'].dt.total_seconds(),
                                mode='lines+markers', name=rival, line=dict(color='red')),
                      row=1, col=1)
    for pit in pit_stops:
        fig.add_vline(x=pit, line=dict(color='green', dash='dash'), row=1, col=1)
    
    # Tire Wear
    fig.add_trace(go.Scatter(x=simulated_laps['LapNumber'], y=simulated_laps['TireWear'],
                            mode='lines', name='Tire Wear (%)', line=dict(color='orange')),
                  row=1, col=2)
    
    # Track Map (simplified)
    track_length = TRACK_LENGTHS.get(grand_prix, 5.0)
    track_x = np.linspace(0, track_length, len(simulated_laps))
    track_y = np.sin(track_x * 2) * 0.5
    fig.add_trace(go.Scatter(x=track_x, y=track_y, mode='lines', name='Track', line=dict(color='gray')), 
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=track_x, y=track_y, mode='markers',
                            marker=dict(size=8, color=simulated_laps['AdjustedLapTime'].dt.total_seconds(),
                                       colorscale='Viridis'), name='Lap Time'),
                  row=2, col=1)
    
    # Stint Breakdown
    stint_laps = [f"Stint {i+1}: {s['laps'][0]}-{s['laps'][-1]}" for i, s in enumerate(stints)]
    fig.add_trace(go.Bar(x=stint_laps, y=[s['avg_time'] for s in stints], name='Avg Lap Time (s)',
                        marker_color='purple'), row=2, col=2)
    
    # Weather Impact
    weather_effect = simulated_laps['AdjustedLapTime'].dt.total_seconds() / simulated_laps['LapTime'].dt.total_seconds()
    fig.add_trace(go.Scatter(x=simulated_laps['LapNumber'], y=weather_effect, mode='lines', 
                            name='Weather Effect', line=dict(color='cyan')), row=3, col=1)
    
    # Strategy Comparison
    pit_configs, times = zip(*results) if 'results' in globals() else ([], [])
    fig.add_trace(go.Scatter(x=[str(cfg) for cfg in pit_configs], 
                            y=[t.total_seconds() / 60 for t in times],
                            mode='lines+markers', name='Total Time (min)', line=dict(color='green')),
                  row=3, col=2)
    fig.add_trace(go.Scatter(x=[str(best_strategy)], y=[total_time.total_seconds() / 60],
                            mode='markers', marker=dict(size=12, color='red'), name='Best'),
                  row=3, col=2)
    
    fig.update_layout(height=900, width=1200, 
                      title_text=f"{driver} vs {rival or 'None'} - {year} {grand_prix} Grand Prix",
                      showlegend=True)
    fig.show()

# Main execution with user input
def main():
    print("F1 Strategy Simulator - Accurate Edition")
    available_years = range(2018, 2025)  # fastf1 data range
    
    # Year selection
    while True:
        year = input(f"Enter year ({min(available_years)}-{max(available_years)}): ").strip()
        try:
            year = int(year)
            if year not in available_years:
                raise ValueError(f"Year must be between {min(available_years)} and {max(available_years)}")
            break
        except ValueError as e:
            print(e if "between" in str(e) else "Invalid year. Enter a number.")
    
    # Race selection
    available_races = get_available_races(year)
    if not available_races:
        print("No races found for this year. Exiting.")
        return
    print(f"Available races for {year}: {', '.join(available_races.keys())}")
    while True:
        grand_prix = input("Enter Grand Prix name (e.g., Monaco): ").strip()
        if grand_prix not in available_races:
            print("Invalid race. Choose from the list above.")
            continue
        break
    
    # Driver selection
    print("Examples: VER (Verstappen), HAM (Hamilton), LEC (Leclerc), NOR (Norris), RUS (Russell)")
    while True:
        driver = input("Enter first driver code: ").strip().upper()
        rival = input("Enter second driver code (or Enter for none): ").strip().upper() or None
        driver_laps, rival_laps, telemetry, weather, track_length = load_race_data(year, grand_prix, driver, rival)
        if driver_laps is not None:
            break
        print("Invalid driver(s). Try again.")
    
    # Process data
    driver_laps = calculate_tire_wear(driver_laps, weather, track_length)
    driver_laps = calculate_performance_factors(driver_laps, track_length)
    if rival_laps is not None:
        rival_laps = calculate_tire_wear(rival_laps, weather, track_length)
        rival_laps = calculate_performance_factors(rival_laps, track_length)
    
    # Simulate race with realistic initial strategy
    total_laps = len(driver_laps)
    initial_pits = [total_laps // 3, 2 * total_laps // 3] if total_laps > 50 else [total_laps // 2]
    simulated_laps, total_time, stints = simulate_race(driver_laps, weather, track_length, initial_pits)
    if rival_laps is not None:
        rival_pits = [total_laps // 2] if total_laps > 50 else [total_laps // 3]
        rival_sim, rival_time, _ = simulate_race(rival_laps, weather, track_length, rival_pits)
        print(f"{driver} Time: {total_time}, {rival} Time: {rival_time}")
    
    # Optimize strategy
    global results
    best_strategy, best_time, results = optimize_strategy(driver_laps, weather, track_length)
    print(f"Best Strategy for {driver}: Pits at {best_strategy}, Time: {best_time}")
    
    # Dashboard
    create_dashboard(simulated_laps, rival_sim if rival else None, initial_pits, stints, 
                    driver, rival, grand_prix, year, best_strategy, total_time)

if __name__ == "__main__":
    main()