"""
Generate synthetic pet compatibility data for bootstrapping ML models.
This creates realistic training data based on expert pet behavior knowledge.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

# Pet breed data with characteristics
BREEDS = {
    'Golden Retriever': {'size': 'large', 'energy': 'high', 'temperament': 'friendly', 'social': 0.9},
    'Labrador': {'size': 'large', 'energy': 'high', 'temperament': 'friendly', 'social': 0.9},
    'German Shepherd': {'size': 'large', 'energy': 'high', 'temperament': 'protective', 'social': 0.6},
    'Poodle': {'size': 'medium', 'energy': 'medium', 'temperament': 'intelligent', 'social': 0.7},
    'Bulldog': {'size': 'medium', 'energy': 'low', 'temperament': 'calm', 'social': 0.7},
    'Beagle': {'size': 'medium', 'energy': 'high', 'temperament': 'friendly', 'social': 0.8},
    'Chihuahua': {'size': 'small', 'energy': 'medium', 'temperament': 'nervous', 'social': 0.4},
    'Corgi': {'size': 'small', 'energy': 'medium', 'temperament': 'friendly', 'social': 0.7},
    'Husky': {'size': 'large', 'energy': 'high', 'temperament': 'energetic', 'social': 0.7},
    'Dachshund': {'size': 'small', 'energy': 'medium', 'temperament': 'stubborn', 'social': 0.5},
    'Boxer': {'size': 'large', 'energy': 'high', 'temperament': 'playful', 'social': 0.8},
    'Shih Tzu': {'size': 'small', 'energy': 'low', 'temperament': 'calm', 'social': 0.6},
    'Rottweiler': {'size': 'large', 'energy': 'medium', 'temperament': 'protective', 'social': 0.5},
    'Pomeranian': {'size': 'small', 'energy': 'medium', 'temperament': 'bold', 'social': 0.5},
    'Border Collie': {'size': 'medium', 'energy': 'high', 'temperament': 'intelligent', 'social': 0.7},
}

SIZE_MAP = {'small': 1, 'medium': 2, 'large': 3}
ENERGY_MAP = {'low': 1, 'medium': 2, 'high': 3}

TEMPERAMENTS = ['friendly', 'calm', 'energetic', 'protective', 'nervous', 'playful', 'intelligent', 'stubborn', 'bold']


def calculate_expert_compatibility(pet1: Dict, pet2: Dict) -> float:
    """
    Calculate compatibility score based on expert pet behavior rules.
    Returns a score between 0 and 1.
    """
    score = 0.5  # baseline

    # Energy level matching (very important) +/- 0.25
    energy_diff = abs(ENERGY_MAP[pet1['energy']] - ENERGY_MAP[pet2['energy']])
    if energy_diff == 0:
        score += 0.25
    elif energy_diff == 1:
        score += 0.1
    else:
        score -= 0.15

    # Size compatibility (important for safety) +/- 0.2
    size_diff = abs(SIZE_MAP[pet1['size']] - SIZE_MAP[pet2['size']])
    if size_diff == 0:
        score += 0.15
    elif size_diff == 1:
        score += 0.05
    else:
        score -= 0.2  # Large dogs with small dogs can be risky

    # Temperament matching +/- 0.25
    if pet1['temperament'] == 'aggressive' or pet2['temperament'] == 'aggressive':
        score -= 0.4
    elif pet1['temperament'] == 'nervous' and pet2['temperament'] == 'energetic':
        score -= 0.2
    elif pet1['temperament'] == 'friendly' and pet2['temperament'] == 'friendly':
        score += 0.25
    elif pet1['temperament'] == 'playful' and pet2['temperament'] == 'playful':
        score += 0.2
    elif pet1['temperament'] == 'calm' and pet2['temperament'] == 'calm':
        score += 0.15

    # Age compatibility +/- 0.15
    age_diff = abs(pet1['age'] - pet2['age'])
    if age_diff < 2:
        score += 0.15
    elif age_diff < 4:
        score += 0.05
    elif age_diff > 8:
        score -= 0.1

    # Social scores +/- 0.15
    avg_social = (pet1['social'] + pet2['social']) / 2
    if avg_social > 0.8:
        score += 0.15
    elif avg_social < 0.5:
        score -= 0.15

    # Clip to valid range
    return np.clip(score, 0.0, 1.0)


def generate_pet_profile() -> Dict:
    """Generate a random pet profile"""
    breed = np.random.choice(list(BREEDS.keys()))
    breed_info = BREEDS[breed]

    # Add some variation to breed characteristics
    age = np.random.randint(1, 15)

    # Slightly vary energy and social scores
    energy_variation = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
    social_score = np.clip(breed_info['social'] + np.random.normal(0, 0.1), 0, 1)

    return {
        'breed': breed,
        'size': breed_info['size'],
        'energy': energy_variation if np.random.random() > 0.7 else breed_info['energy'],
        'temperament': breed_info['temperament'],
        'age': age,
        'social': social_score,
        'weight': {
            'small': np.random.randint(5, 25),
            'medium': np.random.randint(25, 60),
            'large': np.random.randint(60, 100)
        }[breed_info['size']],
    }


def generate_compatibility_dataset(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic compatibility dataset"""

    print(f"Generating {n_samples} synthetic pet pair samples...")

    data = []

    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"  Generated {i} samples...")

        pet1 = generate_pet_profile()
        pet2 = generate_pet_profile()

        # Calculate compatibility score
        score = calculate_expert_compatibility(pet1, pet2)

        # Add some noise to make it more realistic
        score = np.clip(score + np.random.normal(0, 0.05), 0, 1)

        # Create feature row
        row = {
            # Pet 1 features
            'pet1_breed': pet1['breed'],
            'pet1_size': pet1['size'],
            'pet1_energy': pet1['energy'],
            'pet1_temperament': pet1['temperament'],
            'pet1_age': pet1['age'],
            'pet1_social': pet1['social'],
            'pet1_weight': pet1['weight'],

            # Pet 2 features
            'pet2_breed': pet2['breed'],
            'pet2_size': pet2['size'],
            'pet2_energy': pet2['energy'],
            'pet2_temperament': pet2['temperament'],
            'pet2_age': pet2['age'],
            'pet2_social': pet2['social'],
            'pet2_weight': pet2['weight'],

            # Target
            'compatibility_score': score,
        }

        data.append(row)

    df = pd.DataFrame(data)
    print(f"\n✅ Generated {len(df)} samples")
    print(f"Score distribution: mean={df['compatibility_score'].mean():.3f}, std={df['compatibility_score'].std():.3f}")

    return df


def generate_energy_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """Generate synthetic energy state dataset"""

    print(f"\nGenerating {n_samples} synthetic energy state samples...")

    data = []

    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"  Generated {i} samples...")

        # Pet characteristics
        age = np.random.randint(1, 15)
        breed = np.random.choice(list(BREEDS.keys()))
        breed_info = BREEDS[breed]
        base_energy = ENERGY_MAP[breed_info['energy']]

        # Recent activity features
        hours_since_last_activity = np.random.exponential(6)  # Average 6 hours
        total_distance_24h = np.random.gamma(2, 2) * 1000  # meters
        total_duration_24h = np.random.gamma(3, 20)  # minutes
        num_activities_24h = np.random.poisson(2)

        # Time features
        hour_of_day = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)

        # Calculate energy state
        # High activity recently -> low energy
        # Low activity recently -> high energy (rested)
        activity_factor = (total_distance_24h / 5000) + (total_duration_24h / 120)
        time_factor = hours_since_last_activity / 12

        energy_score = base_energy / 3  # normalize to 0-1
        energy_score -= activity_factor * 0.3  # Tired from activity
        energy_score += time_factor * 0.2  # Rested over time

        # Morning boost, evening dip
        if 6 <= hour_of_day <= 10:
            energy_score += 0.15
        elif 20 <= hour_of_day <= 23:
            energy_score -= 0.15

        # Age factor
        if age < 2:
            energy_score += 0.1  # Puppies have lots of energy
        elif age > 10:
            energy_score -= 0.2  # Senior dogs tire easier

        energy_score = np.clip(energy_score, 0, 1)

        # Classify into states
        if energy_score < 0.33:
            energy_state = 'low'
            state_class = 0
        elif energy_score < 0.66:
            energy_state = 'medium'
            state_class = 1
        else:
            energy_state = 'high'
            state_class = 2

        row = {
            'age': age,
            'breed': breed,
            'base_energy_level': breed_info['energy'],
            'hours_since_last_activity': hours_since_last_activity,
            'total_distance_24h': total_distance_24h,
            'total_duration_24h': total_duration_24h,
            'num_activities_24h': num_activities_24h,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'energy_state': energy_state,
            'energy_state_class': state_class,
            'energy_score': energy_score,
        }

        data.append(row)

    df = pd.DataFrame(data)
    print(f"\n✅ Generated {len(df)} energy samples")
    print(f"Class distribution:")
    print(df['energy_state'].value_counts())

    return df


if __name__ == '__main__':
    # Generate compatibility data
    compatibility_df = generate_compatibility_dataset(10000)
    compatibility_df.to_csv('ml/data/compatibility_synthetic.csv', index=False)
    print(f"\n✅ Saved compatibility data to ml/data/compatibility_synthetic.csv")

    # Generate energy data
    energy_df = generate_energy_dataset(5000)
    energy_df.to_csv('ml/data/energy_synthetic.csv', index=False)
    print(f"✅ Saved energy data to ml/data/energy_synthetic.csv")

    # Save breed encodings
    breed_encoding = {breed: idx for idx, breed in enumerate(BREEDS.keys())}
    with open('ml/data/breed_encoding.json', 'w') as f:
        json.dump(breed_encoding, f, indent=2)
    print(f"✅ Saved breed encodings to ml/data/breed_encoding.json")

    print("\n🎉 Synthetic data generation complete!")
