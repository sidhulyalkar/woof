"""
Generate synthetic social graph data for GNN training

Creates realistic pet social networks with:
- Pet nodes with features
- Friendship edges with interaction history
- Community structure (neighborhoods, dog parks)
- Temporal interaction patterns
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
import networkx as nx
from datetime import datetime, timedelta

# Reuse breed data
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


def generate_pet_nodes(n_pets: int = 500) -> pd.DataFrame:
    """Generate pet nodes with features"""
    pets = []

    for i in range(n_pets):
        breed = np.random.choice(list(BREEDS.keys()))
        breed_info = BREEDS[breed]

        age = np.random.randint(1, 15)
        energy_variation = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
        social_score = np.clip(breed_info['social'] + np.random.normal(0, 0.1), 0, 1)

        # Activity level based on recent activities
        activity_level = np.random.beta(2, 2)  # 0-1 scale

        pet = {
            'pet_id': i,
            'breed': breed,
            'size': breed_info['size'],
            'energy': energy_variation if np.random.random() > 0.7 else breed_info['energy'],
            'temperament': breed_info['temperament'],
            'age': age,
            'social_score': social_score,
            'weight': {
                'small': np.random.randint(5, 25),
                'medium': np.random.randint(25, 60),
                'large': np.random.randint(60, 100)
            }[breed_info['size']],
            'activity_level': activity_level,
            # Location (for community structure)
            'location_lat': np.random.normal(37.7749, 0.05),  # SF Bay Area
            'location_lon': np.random.normal(-122.4194, 0.05),
        }

        pets.append(pet)

    return pd.DataFrame(pets)


def generate_social_graph(
    pets_df: pd.DataFrame,
    avg_friends: int = 10,
    community_bonus: float = 0.3
) -> Tuple[List[Tuple[int, int]], pd.DataFrame]:
    """
    Generate social graph with realistic structure

    Uses:
    - Preferential attachment (popular pets get more friends)
    - Homophily (similar pets connect more)
    - Geographic proximity (nearby pets connect more)
    - Community structure (neighborhoods)
    """
    n_pets = len(pets_df)
    edges = []
    edge_features = []

    # Create networkx graph for community detection
    G = nx.Graph()
    G.add_nodes_from(range(n_pets))

    # Track degrees for preferential attachment
    degrees = np.zeros(n_pets)

    for i in range(n_pets):
        pet_i = pets_df.iloc[i]

        # Target number of friends (Poisson distribution)
        target_friends = min(np.random.poisson(avg_friends), n_pets - 1)

        # Calculate connection probabilities
        probs = np.zeros(n_pets)

        for j in range(n_pets):
            if i == j:
                continue

            pet_j = pets_df.iloc[j]

            # Base probability
            prob = 0.01

            # Preferential attachment (popular pets attract more friends)
            prob += 0.02 * (degrees[j] / (avg_friends + 1))

            # Homophily - similar pets connect more
            # Same size
            if pet_i['size'] == pet_j['size']:
                prob += 0.15

            # Similar energy
            energy_diff = abs(ENERGY_MAP[pet_i['energy']] - ENERGY_MAP[pet_j['energy']])
            prob += 0.1 * (1 - energy_diff / 2)

            # Same temperament
            if pet_i['temperament'] == pet_j['temperament']:
                prob += 0.1

            # Age proximity
            age_diff = abs(pet_i['age'] - pet_j['age'])
            prob += 0.05 * max(0, 1 - age_diff / 10)

            # Social compatibility
            prob += 0.1 * (pet_i['social_score'] + pet_j['social_score']) / 2

            # Geographic proximity (community structure)
            dist = np.sqrt(
                (pet_i['location_lat'] - pet_j['location_lat'])**2 +
                (pet_i['location_lon'] - pet_j['location_lon'])**2
            )
            if dist < 0.01:  # Very close (~1km)
                prob += community_bonus

            probs[j] = prob

        # Normalize probabilities
        probs = probs / probs.sum()

        # Sample friends
        if target_friends > 0:
            friend_indices = np.random.choice(
                n_pets,
                size=min(target_friends, n_pets - 1),
                replace=False,
                p=probs
            )

            for j in friend_indices:
                if i < j:  # Avoid duplicates
                    # Generate edge features (interaction history)
                    interactions_count = np.random.poisson(5) + 1
                    last_interaction_days = np.random.exponential(7)  # Last interaction
                    avg_duration = np.random.gamma(3, 10)  # Minutes

                    edges.append((i, j))
                    edge_features.append({
                        'pet_a': i,
                        'pet_b': j,
                        'interactions_count': interactions_count,
                        'last_interaction_days': last_interaction_days,
                        'avg_duration_minutes': avg_duration,
                    })

                    # Update degrees
                    degrees[i] += 1
                    degrees[j] += 1

                    # Add to networkx graph
                    G.add_edge(i, j)

    print(f"Generated {len(edges)} edges")
    print(f"Average degree: {degrees.mean():.2f}")
    print(f"Max degree: {int(degrees.max())}")

    # Detect communities
    communities = nx.community.louvain_communities(G)
    print(f"Detected {len(communities)} communities")

    # Add community labels to pets
    community_labels = np.zeros(n_pets, dtype=int)
    for comm_id, community in enumerate(communities):
        for pet_id in community:
            community_labels[pet_id] = comm_id

    pets_df['community'] = community_labels

    return edges, pd.DataFrame(edge_features)


def generate_compatibility_labels(
    pets_df: pd.DataFrame,
    edges: List[Tuple[int, int]],
    edge_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate ground truth compatibility scores for connected pets

    Based on:
    - Interaction frequency (high interactions = high compatibility)
    - Recency (recent interactions = still compatible)
    - Duration (long playdates = high compatibility)
    """
    labels = []

    for idx, (i, j) in enumerate(edges):
        edge_feat = edge_features_df.iloc[idx]

        # Base score from interactions
        interaction_score = min(edge_feat['interactions_count'] / 10, 1.0)

        # Recency bonus (recent = better)
        recency_score = np.exp(-edge_feat['last_interaction_days'] / 30)

        # Duration bonus
        duration_score = min(edge_feat['avg_duration_minutes'] / 60, 1.0)

        # Combined score
        compatibility = (
            0.5 * interaction_score +
            0.3 * recency_score +
            0.2 * duration_score
        )

        # Add noise
        compatibility = np.clip(compatibility + np.random.normal(0, 0.1), 0, 1)

        labels.append({
            'pet_a': i,
            'pet_b': j,
            'compatibility_score': compatibility,
        })

    return pd.DataFrame(labels)


def main():
    print("Generating synthetic social graph data for GNN training...\n")

    # Generate pets
    print("1. Generating pet nodes...")
    pets_df = generate_pet_nodes(n_pets=500)
    print(f"   Created {len(pets_df)} pets")

    # Generate social graph
    print("\n2. Generating social graph...")
    edges, edge_features_df = generate_social_graph(pets_df, avg_friends=10)

    # Generate compatibility labels
    print("\n3. Generating compatibility labels...")
    labels_df = generate_compatibility_labels(pets_df, edges, edge_features_df)
    print(f"   Created {len(labels_df)} compatibility labels")

    # Save data
    print("\n4. Saving data...")
    pets_df.to_csv('ml/data/graph_pets.csv', index=False)
    edge_features_df.to_csv('ml/data/graph_edges.csv', index=False)
    labels_df.to_csv('ml/data/graph_labels.csv', index=False)

    # Save edge list for PyTorch Geometric
    edge_list = np.array(edges).T
    np.save('ml/data/graph_edge_index.npy', edge_list)

    print("\n✅ Social graph data generated successfully!")
    print(f"   Pets: ml/data/graph_pets.csv")
    print(f"   Edges: ml/data/graph_edges.csv")
    print(f"   Labels: ml/data/graph_labels.csv")
    print(f"   Edge Index: ml/data/graph_edge_index.npy")

    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Nodes: {len(pets_df)}")
    print(f"   Edges: {len(edges)}")
    print(f"   Avg compatibility: {labels_df['compatibility_score'].mean():.3f}")
    print(f"   Communities: {pets_df['community'].nunique()}")


if __name__ == '__main__':
    main()
