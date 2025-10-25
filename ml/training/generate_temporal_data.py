"""
Generate synthetic temporal activity sequences for Transformer training

Creates realistic activity patterns with:
- Daily routines (morning walks, evening play)
- Weekly patterns (weekends vs weekdays)
- Seasonal variations
- Pet-specific preferences
- Owner schedules
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ACTIVITY_TYPES = {
    0: 'walk',
    1: 'run',
    2: 'play',
    3: 'training',
    4: 'rest',
    5: 'social',  # meeting other pets
    6: 'fetch',
    7: 'swim',
    8: 'hike',
    9: 'agility',
}

def generate_pet_schedule(
    pet_id: int,
    n_days: int = 90,
    base_energy: str = 'medium'
) -> pd.DataFrame:
    """Generate activity sequence for a single pet"""

    activities = []
    current_date = datetime.now() - timedelta(days=n_days)

    # Pet-specific preferences
    morning_person = np.random.random() > 0.5
    weekend_warrior = np.random.random() > 0.6
    social_butterfly = np.random.random() > 0.4

    # Activity probabilities by time of day
    activity_prefs = {
        'morning': [0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01],  # Mostly walks
        'midday': [0.2, 0.05, 0.3, 0.1, 0.2, 0.1, 0.02, 0.01, 0.01, 0.01],  # Play, rest
        'afternoon': [0.3, 0.15, 0.2, 0.05, 0.1, 0.15, 0.02, 0.01, 0.01, 0.01],  # Mix
        'evening': [0.25, 0.05, 0.35, 0.05, 0.15, 0.1, 0.02, 0.01, 0.01, 0.01],  # Play
    }

    # Energy levels
    energy_map = {'low': 0.6, 'medium': 1.0, 'high': 1.4}
    energy_multiplier = energy_map[base_energy]

    for day in range(n_days):
        date = current_date + timedelta(days=day)
        is_weekend = date.weekday() >= 5

        # Number of activities per day (more on weekends for weekend warriors)
        if is_weekend and weekend_warrior:
            n_activities = np.random.poisson(4) + 2
        else:
            n_activities = np.random.poisson(2) + 1

        n_activities = int(n_activities * energy_multiplier)

        # Generate activities for this day
        for _ in range(n_activities):
            # Choose time of day
            if morning_person and np.random.random() > 0.3:
                hour = np.random.randint(6, 10)
                time_period = 'morning'
            elif np.random.random() > 0.5:
                hour = np.random.randint(14, 18)
                time_period = 'afternoon'
            else:
                hour = np.random.randint(18, 21)
                time_period = 'evening'

            # Choose activity type based on time of day
            activity_type = np.random.choice(
                len(ACTIVITY_TYPES),
                p=activity_prefs[time_period]
            )

            # Activity features
            if activity_type == 0:  # walk
                duration = np.random.gamma(3, 10)  # ~30 min
                distance = duration * 70  # ~70m/min walking pace
                intensity = 0.4 + np.random.random() * 0.2
            elif activity_type == 1:  # run
                duration = np.random.gamma(2, 8)  # ~16 min
                distance = duration * 150  # ~150m/min running
                intensity = 0.7 + np.random.random() * 0.2
            elif activity_type == 2:  # play
                duration = np.random.gamma(2, 10)  # ~20 min
                distance = duration * 30  # less distance, more intensity
                intensity = 0.6 + np.random.random() * 0.3
            elif activity_type == 3:  # training
                duration = np.random.gamma(1.5, 8)  # ~12 min
                distance = duration * 20
                intensity = 0.5 + np.random.random() * 0.2
            elif activity_type == 4:  # rest
                duration = np.random.gamma(5, 10)  # ~50 min
                distance = 0
                intensity = 0.1
            elif activity_type == 5:  # social
                duration = np.random.gamma(3, 15)  # ~45 min
                distance = duration * 40
                intensity = 0.5 + np.random.random() * 0.3
            else:  # other activities
                duration = np.random.gamma(2, 10)
                distance = duration * 50
                intensity = 0.5 + np.random.random() * 0.3

            # Calories burned (rough estimate)
            calories = duration * intensity * 5

            # Social context
            with_owner = np.random.random() > 0.2
            with_other_pets = social_butterfly and np.random.random() > 0.6

            activity = {
                'pet_id': pet_id,
                'timestamp': date.replace(hour=hour, minute=np.random.randint(0, 60)),
                'activity_type': activity_type,
                'duration_minutes': duration,
                'distance_meters': distance,
                'intensity': intensity,
                'calories': calories,
                'hour_of_day': hour,
                'day_of_week': date.weekday(),
                'is_weekend': is_weekend,
                'with_owner': with_owner,
                'with_other_pets': with_other_pets,
            }

            activities.append(activity)

    df = pd.DataFrame(activities)
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def create_sequences(
    activities_df: pd.DataFrame,
    seq_length: int = 20,
    prediction_horizon: int = 1
) -> pd.DataFrame:
    """
    Create input/output sequences for Transformer training

    Input: Last seq_length activities
    Output: Next activity (type, duration, optimal time)
    """
    sequences = []

    for pet_id in activities_df['pet_id'].unique():
        pet_activities = activities_df[activities_df['pet_id'] == pet_id].reset_index(drop=True)

        if len(pet_activities) < seq_length + prediction_horizon:
            continue

        # Create sequences with sliding window
        for i in range(len(pet_activities) - seq_length - prediction_horizon + 1):
            # Input sequence
            input_seq = pet_activities.iloc[i:i+seq_length]

            # Target (next activity)
            target = pet_activities.iloc[i+seq_length:i+seq_length+prediction_horizon]

            if len(target) < prediction_horizon:
                continue

            target_activity = target.iloc[0]

            # Energy state (inferred from recent activity)
            recent_intensity = input_seq['intensity'].tail(5).mean()
            recent_duration = input_seq['duration_minutes'].tail(5).sum()

            if recent_intensity > 0.6 and recent_duration > 60:
                energy_before = 'low'  # Tired from recent activity
            elif recent_intensity < 0.3 or recent_duration < 20:
                energy_before = 'high'  # Rested
            else:
                energy_before = 'medium'

            sequence = {
                'pet_id': pet_id,
                'sequence_id': f"{pet_id}_{i}",

                # Input features (arrays)
                'input_activity_types': input_seq['activity_type'].tolist(),
                'input_durations': input_seq['duration_minutes'].tolist(),
                'input_distances': input_seq['distance_meters'].tolist(),
                'input_intensities': input_seq['intensity'].tolist(),
                'input_calories': input_seq['calories'].tolist(),
                'input_hours': input_seq['hour_of_day'].tolist(),
                'input_days': input_seq['day_of_week'].tolist(),
                'input_with_owner': input_seq['with_owner'].tolist(),

                # Target
                'target_activity_type': int(target_activity['activity_type']),
                'target_duration': target_activity['duration_minutes'],
                'target_hour': int(target_activity['hour_of_day']),

                # Energy states
                'energy_before': energy_before,
                'energy_after': 'low' if target_activity['intensity'] > 0.6 else 'medium',
            }

            sequences.append(sequence)

    return pd.DataFrame(sequences)


def main():
    print("Generating synthetic temporal activity sequences...\n")

    # Generate activities for multiple pets
    print("1. Generating activity sequences...")
    all_activities = []

    for pet_id in range(100):  # 100 pets
        if pet_id % 20 == 0:
            print(f"   Generated {pet_id} pets...")

        # Random energy level
        base_energy = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.6, 0.2])

        pet_activities = generate_pet_schedule(pet_id, n_days=90, base_energy=base_energy)
        all_activities.append(pet_activities)

    activities_df = pd.concat(all_activities, ignore_index=True)
    print(f"   Created {len(activities_df)} activities across 100 pets")

    # Create training sequences
    print("\n2. Creating training sequences...")
    sequences_df = create_sequences(activities_df, seq_length=20)
    print(f"   Created {len(sequences_df)} training sequences")

    # Save data
    print("\n3. Saving data...")
    activities_df.to_csv('ml/data/temporal_activities.csv', index=False)
    sequences_df.to_csv('ml/data/temporal_sequences.csv', index=False)

    print("\n✅ Temporal activity data generated successfully!")
    print(f"   Activities: ml/data/temporal_activities.csv")
    print(f"   Sequences: ml/data/temporal_sequences.csv")

    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Total activities: {len(activities_df)}")
    print(f"   Training sequences: {len(sequences_df)}")
    print(f"   Activity type distribution:")
    for act_id, act_name in ACTIVITY_TYPES.items():
        count = len(activities_df[activities_df['activity_type'] == act_id])
        pct = 100 * count / len(activities_df)
        print(f"      {act_name}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
