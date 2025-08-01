# PetPath Data Models

## Overview

This document describes the core data models used in the PetPath platform.

## Relational Database Schema (PostgreSQL)

### Core Tables

#### Users
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    avatar_url TEXT,
    location POINT,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Pets
```sql
CREATE TABLE pets (
    id UUID PRIMARY KEY,
    owner_id UUID NOT NULL REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    breed VARCHAR(255) NOT NULL,
    age INTEGER NOT NULL,
    size VARCHAR(50) NOT NULL,
    device_id VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Friendships
```sql
CREATE TABLE friendships (
    id UUID PRIMARY KEY,
    pet_a_id UUID NOT NULL REFERENCES pets(id),
    pet_b_id UUID NOT NULL REFERENCES pets(id),
    status VARCHAR(50) NOT NULL,
    compatibility_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Social Content

#### Posts
```sql
CREATE TABLE posts (
    id UUID PRIMARY KEY,
    author_user_id UUID NOT NULL REFERENCES users(id),
    pet_id UUID NOT NULL REFERENCES pets(id),
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Activities and Fitness

#### Activities
```sql
CREATE TABLE activities (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    pet_id UUID NOT NULL REFERENCES pets(id),
    activity_type VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    distance_km DECIMAL(8,3),
    steps_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Locations and Meetups

#### Location Entities
```sql
CREATE TABLE location_entities (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    coordinates POINT NOT NULL,
    amenities TEXT[],
    rating DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Graph Database Schema (Neo4j)

### Node Labels
- `(:User)` - User profiles
- `(:Pet)` - Pet profiles
- `(:Location)` - Pet-friendly locations
- `(:Activity)` - Activity tracking

### Relationship Types
- `[:OWNS]` - User owns pet
- `[:FRIENDS_WITH]` - Pet friendships
- `[:VISITED]` - Pet visited location
- `[:PARTICIPATED_IN]` - Pet in activity
- `[:INTERACTED_WITH]` - Pet interactions

This simplified model supports the core PetPath functionality including social networking, activity tracking, and meetup coordination.