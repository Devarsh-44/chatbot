import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import random

# Enable caching
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

# Track lengths (in km)
TRACK_LENGTHS = {
    'Bahrain': 5.4, 'Saudi Arabia': 6.2, 'Australia': 5.3, 'Azerbaijan': 6.0, 'Miami': 5.4,
    'Monaco': 3.3, 'Spain': 4.7, 'Canada': 4.4, 'Austria': 4.3, 'Great Britain': 5.9,
    'Hungary': 4.4, 'Belgium': 7.0, 'Netherlands': 4.3, 'Italy': 5.8, 'Singapore': 5.1,
    'Japan': 5.8, 'Qatar': 5.4, 'United States': 5.5, 'Mexico': 4.3, 'Brazil': 4.3,
    'Las Vegas': 6.2, 'Abu Dhabi': 5.3
}

# Tire colors and wear rates
TIRE_COLORS = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white'}
TIRE_WEAR_RATES = {'SOFT': 0.03, 'MEDIUM': 0.02, 'HARD': 0.01}  # % wear per km

# Get available races
def get_available_races(year):
    try:
        events = fastf1.get_event_schedule(year)
        return {row['EventName'].replace(" Grand Prix", ""): row['RoundNumber'] 
                for _, row in events.iterrows() if row['Session5'] == 'Race'}
    except:
        print("🏁 Couldn’t load races, but we’ll roll with it!")
        return TRACK_LENGTHS

# Load race data
def load_race_data(year, grand_prix, driver, rival):
    try:
        session = fastf1.get_session(year, grand_prix, 'R')
        session.load(telemetry=True, laps=True, weather=True)
        driver_laps = session.laps.pick_driver(driver)
        rival_laps = session.laps.pick_driver(rival) if rival else None
        
        if driver_laps.empty or (rival and (rival_laps is None or rival_laps.empty)):
            raise ValueError("Driver not found!")
        
        weather = session.weather_data
        track_length = TRACK_LENGTHS.get(grand_prix, 5.0)
        print(f"🚗 Loaded {len(driver_laps)} laps for {driver}" + 
              (f" and {len(rival_laps)} for {rival}" if rival else "") +
              f" at {grand_prix} ({track_length} km per lap) 🎉")
        return driver_laps, rival_laps, weather, track_length, session
    except Exception as e:
        print(f"⚠️ Oops: {e}")
        return None, None, None, None, None

# Tire wear with user-selected compounds
def calculate_tire_wear(laps, weather, track_length, pit_stops, tire_choices):
    if laps is None or weather is None:
        return None, None
    
    rain = weather['Rainfall'].mean() > 0.1
    tire_types = tire_choices
    
    laps = laps.copy()
    laps['TireWear'] = 0.0
    stint_start = 0
    stint_num = 0
    
    for pit_lap in sorted(pit_stops + [len(laps) + 1]):
        if pit_lap <= len(laps):
            stint_end = laps.index[laps['LapNumber'] == pit_lap][0] if pit_lap <= len(laps) else laps.index[-1] + 1
            stint = laps.iloc[stint_start:stint_end]
            if not stint.empty:
                tire_type = tire_types[stint_num] if stint_num < len(tire_types) else 'MEDIUM'
                wear_rate = TIRE_WEAR_RATES.get(tire_type, 0.02) * track_length * (1.2 if rain else 1.0)
                laps.loc[stint.index, 'TireWear'] = np.linspace(0, wear_rate * len(stint), len(stint))
                stint_num += 1
            stint_start = stint_end
    
    laps['TireWear'] = np.clip(laps['TireWear'], 0, 100)
    return laps, tire_types

# Race simulation with commentary
def simulate_race(laps, weather, track_length, pit_stops, driver, tire_choices):
    if laps is None or weather is None:
        return None, None, None, None, None
    
    race_laps = laps.copy()
    race_laps['LapTime'] = pd.to_timedelta(race_laps['LapTime'], errors='coerce').fillna(pd.Timedelta(seconds=90))
    race_laps['AdjustedLapTime'] = race_laps['LapTime']
    
    rain_effect = 1.1 if weather['Rainfall'].mean() > 0.1 else 1.0
    stints = []
    stint_start = 0
    tire_types = tire_choices
    stint_num = 0
    commentary = []
    
    if rain_effect > 1.0:
        commentary.append("☔ Rain is shaking things up out there!")
    
    for i, lap in race_laps.iterrows():
        tire_effect = 1 + 0.005 * (lap['TireWear'] / 100)
        race_laps.loc[i, 'AdjustedLapTime'] *= rain_effect * tire_effect
        if lap['TireWear'] > 80 and random.random() < 0.1:
            commentary.append(f"Lap {int(lap['LapNumber'])}: {driver}’s tires are screaming—time for a pit soon?")
    
    for pit_lap in sorted(pit_stops + [len(race_laps) + 1]):
        if pit_lap <= len(race_laps):
            idx = race_laps.index[race_laps['LapNumber'] == pit_lap][0]
            race_laps.loc[idx, 'AdjustedLapTime'] += pd.Timedelta(seconds=22)
            stint = race_laps.iloc[stint_start:idx]
            if not stint.empty:
                tire_type = tire_types[stint_num] if stint_num < len(tire_types) else 'MEDIUM'
                stints.append({'laps': stint['LapNumber'].tolist(), 'avg_time': stint['AdjustedLapTime'].mean().total_seconds(), 'tire': tire_type})
                commentary.append(f"Lap {pit_lap}: {driver} pits for {tire_type.lower()} tires—great stop! 🛠️")
                stint_num += 1
            race_laps.loc[race_laps['LapNumber'] > pit_lap, 'TireWear'] = np.linspace(
                0, 0.02 * track_length * sum(race_laps['LapNumber'] > pit_lap), sum(race_laps['LapNumber'] > pit_lap))
            stint_start = idx + 1
    
    total_time = race_laps['AdjustedLapTime'].sum()
    commentary.append(f"🏁 {driver} crosses the line after a thrilling race!")
    return race_laps, total_time, stints, tire_types, commentary

# Predict finishing position
def predict_position(total_time, session, driver):
    try:
        # Get real race results
        results = session.results
        winner_time = results[results['Position'] == 1]['Time'].iloc[0].total_seconds()
        if pd.isna(winner_time):  # If no official time (e.g., DNF), estimate from laps
            winner_laps = session.laps.pick_fastest().pick_driver(results[results['Position'] == 1]['DriverCode'].iloc[0])
            winner_time = winner_laps['LapTime'].total_seconds() * len(session.laps.pick_driver(driver))
        
        user_time = total_time.total_seconds()
        # Simulate field: winner + incremental gaps (e.g., 0.5% per position)
        positions = []
        for pos in range(1, 21):
            benchmark_time = winner_time * (1 + 0.005 * (pos - 1))  # 0.5% slower per position
            positions.append((pos, benchmark_time))
        
        # Find user's position
        user_pos = 1
        for pos, time in positions:
            if user_time > time:
                user_pos = pos + 1
            else:
                break
        return min(user_pos, 20)  # Cap at 20th
    except Exception as e:
        print(f"⚠️ Position prediction failed: {e}. Defaulting to a fun guess!")
        return random.randint(1, 20)  # Fallback for robustness

# Find best strategy
def optimize_strategy(laps, weather, track_length, driver):
    if laps is None:
        return None, None
    
    total_laps = len(laps)
    best_time = None
    best_pits = []
    tire_options = ['SOFT', 'MEDIUM', 'HARD']
    
    pit_options = [total_laps // 2] if total_laps < 50 else [total_laps // 3, 2 * total_laps // 3]
    for pits in [[], [pit_options[0]], pit_options]:
        tire_choices = [random.choice(tire_options) for _ in range(len(pits) + 1)]
        laps_with_wear, _ = calculate_tire_wear(laps, weather, track_length, pits, tire_choices)
        _, total_time, _, _, _ = simulate_race(laps_with_wear, weather, track_length, pits, driver, tire_choices)
        if total_time is not None:
            if best_time is None or total_time < best_time:
                best_time = total_time
                best_pits = pits
    return best_pits, best_time

# Fun dashboard
def create_dashboard(simulated_laps, rival_laps, pit_stops, stints, driver, rival, grand_prix, year, tire_types, commentary):
    if simulated_laps is None:
        return
    
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("🏎️ Lap Speeds", "🛞 Tire Life",
                                        "🌍 Race Journey", "🛠️ Pit Stop Story"))
    
    # Lap Speeds
    fig.add_trace(go.Scatter(x=simulated_laps['LapNumber'], 
                            y=simulated_laps['AdjustedLapTime'].dt.total_seconds(),
                            mode='lines+markers', name=f"{driver} 🚗", line=dict(color='blue')),
                  row=1, col=1)
    if rival_laps is not None:
        fig.add_trace(go.Scatter(x=rival_laps['LapNumber'], 
                                y=rival_laps['AdjustedLapTime'].dt.total_seconds(),
                                mode='lines+markers', name=f"{rival} 🚗", line=dict(color='red')),
                      row=1, col=1)
    for pit in pit_stops:
        fig.add_vline(x=pit, line=dict(color='green', dash='dash'), row=1, col=1)
    fig.update_yaxes(title_text="Seconds per Lap ⏱️", row=1, col=1)
    
    # Tire Life
    fig.add_trace(go.Scatter(x=simulated_laps['LapNumber'], y=simulated_laps['TireWear'],
                            mode='lines', name='Wear (%)', line=dict(color='orange')),
                  row=1, col=2)
    fig.update_yaxes(title_text="Tire Wear (%) 🛞", row=1, col=2)
    
    # Race Journey
    track_length = TRACK_LENGTHS.get(grand_prix, 5.0)
    track_x = np.linspace(0, track_length * len(simulated_laps), len(simulated_laps))
    track_y = np.zeros(len(simulated_laps))
    fig.add_trace(go.Scatter(x=track_x, y=track_y, mode='lines+markers',
                            marker=dict(size=10, color=simulated_laps['AdjustedLapTime'].dt.total_seconds(),
                                       colorscale='Viridis'), name='Speed Colors 🌈'),
                  row=2, col=1)
    fig.update_xaxes(title_text="Distance (km) 🌎", row=2, col=1)
    
    # Pit Stop Story
    stint_labels = [f"Part {i+1}: Laps {s['laps'][0]}-{s['laps'][-1]}" for i, s in enumerate(stints)]
    tire_colors = [TIRE_COLORS[s['tire']] for s in stints]
    fig.add_trace(go.Bar(x=stint_labels, y=[s['avg_time'] for s in stints], name='Avg Speed',
                        marker_color=tire_colors), row=2, col=2)
    fig.update_yaxes(title_text="Avg Time (seconds) ⏱️", row=2, col=2)
    
    fig.update_layout(height=700, width=1000, 
                      title_text=f"🏁 {driver} vs {rival or 'Solo Star'} - {year} {grand_prix} Race Party! 🎉",
                      showlegend=True, template='plotly_dark',
                      annotations=[dict(text="\n".join(commentary), xref="paper", yref="paper", x=0.5, y=-0.1, showarrow=False)])
    fig.show()

# Main execution with interactivity
def main():
    print("🎉 Welcome to Your F1 Race Adventure! 🏎️")
    print("Pick a race, drivers, and strategy—see where you finish! 👑")
    
    # Year
    while True:
        year = input("Which year? (2018-2024, like 2023): ").strip()
        try:
            year = int(year)
            if 2018 <= year <= 2024:
                break
            print("⏳ Pick a year between 2018 and 2024!")
        except:
            print("🚫 Type a number, like 2023!")
    
    # Race
    races = get_available_races(year)
    print(f"🏁 Races in {year}: {', '.join(sorted(races.keys()))}")
    while True:
        grand_prix = input("Which race? (e.g., Monaco): ").strip()
        if grand_prix in races:
            break
        print("🤔 That’s not on the list! Try again!")
    
    # Drivers
    print("🌟 Driver codes: VER (Max), HAM (Lewis), NOR (Lando), LEC (Charles)")
    while True:
        driver = input("First driver (e.g., VER): ").strip().upper()
        rival = input("Second driver (or Enter for solo): ").strip().upper() or None
        driver_laps, rival_laps, weather, track_length, session = load_race_data(year, grand_prix, driver, rival)
        if driver_laps is not None:
            break
        print("🚨 Driver not found! Try again!")
    
    # Pit stops and tire selection
    total_laps = len(driver_laps)
    print(f"🏁 This race has {total_laps} laps!")
    pit_stops = []
    tire_choices = []
    stint_count = 1
    
    while True:
        pit = input(f"When should {driver} pit? (1-{total_laps}, or Enter to finish): ").strip()
        if not pit:
            if not pit_stops:
                tire = input("Pick tires for the whole race (S for Soft, M for Medium, H for Hard): ").strip().upper()
                tire_choices.append({'S': 'SOFT', 'M': 'MEDIUM', 'H': 'HARD'}.get(tire, 'MEDIUM'))
            break
        try:
            pit = int(pit)
            if 1 <= pit <= total_laps and pit not in pit_stops:
                pit_stops.append(pit)
                tire = input(f"Pick tires for stint {stint_count} (S for Soft, M for Medium, H for Hard): ").strip().upper()
                tire_choices.append({'S': 'SOFT', 'M': 'MEDIUM', 'H': 'HARD'}.get(tire, 'MEDIUM'))
                print(f"✅ Pit stop at lap {pit} with {tire_choices[-1]} tires!")
                stint_count += 1
            else:
                print("🤔 Invalid or duplicate lap! Try again!")
        except:
            print("🚫 Type a number!")
    
    if pit_stops:
        tire = input(f"Pick tires for stint {stint_count} (S for Soft, M for Medium, H for Hard): ").strip().upper()
        tire_choices.append({'S': 'SOFT', 'M': 'MEDIUM', 'H': 'HARD'}.get(tire, 'MEDIUM'))
    
    # Simulate
    print(f"\n🏎️ Revving up the engines for {driver}" + (f" and {rival}" if rival else "") + "!")
    driver_laps, driver_tires = calculate_tire_wear(driver_laps, weather, track_length, pit_stops, tire_choices)
    simulated_laps, total_time, stints, tire_types, commentary = simulate_race(driver_laps, weather, track_length, pit_stops, driver, tire_choices)
    
    if rival_laps is not None:
        rival_pits = [total_laps // 3]
        rival_tire_choices = ['MEDIUM', 'HARD']
        rival_laps, rival_tires = calculate_tire_wear(rival_laps, weather, track_length, rival_pits, rival_tire_choices)
        rival_sim, rival_time, _, _, rival_commentary = simulate_race(rival_laps, weather, track_length, rival_pits, rival, rival_tire_choices)
        print(f"🎉 {driver} finished in {total_time.total_seconds() // 60:.0f} minutes!")
        print(f"🎉 {rival} finished in {rival_time.total_seconds() // 60:.0f} minutes!")
        commentary.extend(rival_commentary)
    else:
        print(f"🎉 {driver} finished in {total_time.total_seconds() // 60:.0f} minutes!")
    
    # Predict position
    driver_position = predict_position(total_time, session, driver)
    position_suffix = {1: "st", 2: "nd", 3: "rd"}.get(driver_position, "th")
    print(f"🏆 With your strategy, {driver} finishes in {driver_position}{position_suffix} place! Amazing job! 🌟")
    if rival_laps is not None:
        rival_position = predict_position(rival_time, session, rival)
        rival_suffix = {1: "st", 2: "nd", 3: "rd"}.get(rival_position, "th")
        print(f"🏆 {rival} finishes in {rival_position}{rival_suffix} place!")
        if driver_position < rival_position:
            commentary.append(f"🎉 {driver} beats {rival}—what a race!")
        elif driver_position > rival_position:
            commentary.append(f"😬 {rival} takes the edge over {driver} this time!")
        else:
            commentary.append(f"🏁 It’s a tie between {driver} and {rival}—epic battle!")
    
    # Best strategy
    best_pits, best_time = optimize_strategy(driver_laps, weather, track_length, driver)
    print(f"\n🏆 Best plan for {driver}: Pit at {best_pits or 'no stops'} " +
          f"for {best_time.total_seconds() // 60:.0f} minutes!")
    
    # Tire info
    print(f"\n🛞 {driver} used these tires: {', '.join([f'{t} ({TIRE_COLORS[t]})' for t in driver_tires])}")
    if rival_laps is not None:
        print(f"🛞 {rival} used these tires: {', '.join([f'{t} ({TIRE_COLORS[t]})' for t in rival_tires])}")
    
    # Fun fact
    facts = ["F1 cars can go from 0 to 60 mph in under 2 seconds! 🚀",
             "Tires can lose up to 20% grip when worn out! 🛞",
             "A pit stop can take just 2 seconds in real life! ⏱️"]
    print(f"\n🌟 Fun Fact: {random.choice(facts)}")
    
    # Show dashboard with commentary
    print("\n🎙️ Race Highlights:")
    for line in commentary:
        print(f"  {line}")
    print("\n🎨 Here’s your race story in pictures!")
    create_dashboard(simulated_laps, rival_sim if rival_laps is not None else None, pit_stops, stints, 
                    driver, rival, grand_prix, year, tire_types, commentary)

if __name__ == "__main__":
    main()