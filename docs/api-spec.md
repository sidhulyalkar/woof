# PetPath API Specification

## Overview

The PetPath API provides endpoints for a social fitness platform designed for pet owners and their pets. It integrates real-time geographic and activity data from pet wearables and human fitness trackers.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.petpath.com`

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Authentication

#### POST /auth/register
Register a new user account.

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "name": "John Doe",
    "email": "john@example.com"
  }
}
```

#### POST /auth/login
Authenticate a user and receive a JWT token.

**Request Body:**
```json
{
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "name": "John Doe",
    "email": "john@example.com"
  }
}
```

### Users

#### GET /users/me
Get the current user's profile.

**Response:**
```json
{
  "id": "uuid",
  "name": "John Doe",
  "email": "john@example.com",
  "avatar_url": "https://example.com/avatar.jpg",
  "preferences": {
    "pet_size_filter": ["small", "medium"],
    "pet_age_filter": ["adult", "senior"]
  },
  "created_at": "2023-01-01T00:00:00Z"
}
```

#### PATCH /users/me
Update the current user's profile.

**Request Body:**
```json
{
  "name": "John Smith",
  "preferences": {
    "pet_size_filter": ["small", "medium", "large"],
    "pet_age_filter": ["puppy", "adult", "senior"]
  }
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "John Smith",
  "email": "john@example.com",
  "avatar_url": "https://example.com/avatar.jpg",
  "preferences": {
    "pet_size_filter": ["small", "medium", "large"],
    "pet_age_filter": ["puppy", "adult", "senior"]
  },
  "updated_at": "2023-01-02T00:00:00Z"
}
```

### Pets

#### GET /pets
Get all pets belonging to the current user.

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "Buddy",
    "breed": "Golden Retriever",
    "age": 3,
    "avatar_url": "https://example.com/buddy.jpg",
    "device_id": "airtag-123",
    "owner_id": "user-uuid",
    "created_at": "2023-01-01T00:00:00Z"
  }
]
```

#### POST /pets
Add a new pet to the current user's account.

**Request Body:**
```json
{
  "name": "Buddy",
  "breed": "Golden Retriever",
  "age": 3,
  "device_id": "airtag-123"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "Buddy",
  "breed": "Golden Retriever",
  "age": 3,
  "avatar_url": "https://example.com/buddy.jpg",
  "device_id": "airtag-123",
  "owner_id": "user-uuid",
  "created_at": "2023-01-01T00:00:00Z"
}
```

#### PATCH /pets/:id
Update a pet's information.

**Request Body:**
```json
{
  "name": "Buddy Jr.",
  "age": 4
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "Buddy Jr.",
  "breed": "Golden Retriever",
  "age": 4,
  "avatar_url": "https://example.com/buddy.jpg",
  "device_id": "airtag-123",
  "owner_id": "user-uuid",
  "updated_at": "2023-01-02T00:00:00Z"
}
```

#### DELETE /pets/:id
Delete a pet from the current user's account.

**Response:**
```json
{
  "message": "Pet deleted successfully"
}
```

### Social Features

#### POST /friends/request
Send a friend request between two pets.

**Request Body:**
```json
{
  "pet_a_id": "uuid",
  "pet_b_id": "uuid"
}
```

**Response:**
```json
{
  "id": "uuid",
  "pet_a_id": "uuid",
  "pet_b_id": "uuid",
  "status": "pending",
  "compatibility_score": 0.85,
  "created_at": "2023-01-01T00:00:00Z"
}
```

#### POST /friends/respond
Respond to a friend request.

**Request Body:**
```json
{
  "friendship_id": "uuid",
  "status": "accepted"
}
```

**Response:**
```json
{
  "id": "uuid",
  "pet_a_id": "uuid",
  "pet_b_id": "uuid",
  "status": "accepted",
  "compatibility_score": 0.85,
  "updated_at": "2023-01-02T00:00:00Z"
}
```

#### POST /follows
Follow another pet.

**Request Body:**
```json
{
  "follower_pet_id": "uuid",
  "followed_pet_id": "uuid"
}
```

**Response:**
```json
{
  "id": "uuid",
  "follower_pet": "uuid",
  "followed_pet": "uuid",
  "created_at": "2023-01-01T00:00:00Z"
}
```

#### DELETE /follows/:pet_id
Unfollow a pet.

**Response:**
```json
{
  "message": "Unfollowed successfully"
}
```

### Social Feed

#### GET /feed
Get the social feed with optional filtering.

**Query Parameters:**
- `filter`: `all`, `nearby`, `breed` (default: `all`)
- `limit`: Number of posts to return (default: 20)
- `offset`: Offset for pagination (default: 0)

**Response:**
```json
{
  "posts": [
    {
      "id": "uuid",
      "author_user": {
        "id": "uuid",
        "name": "John Doe",
        "avatar_url": "https://example.com/john.jpg"
      },
      "pet": {
        "id": "uuid",
        "name": "Buddy",
        "breed": "Golden Retriever",
        "avatar_url": "https://example.com/buddy.jpg"
      },
      "content": "Great walk in the park today!",
      "media_urls": ["https://example.com/walk.jpg"],
      "created_at": "2023-01-01T00:00:00Z",
      "likes_count": 5,
      "comments_count": 2,
      "liked_by_user": true
    }
  ],
  "total": 50,
  "limit": 20,
  "offset": 0
}
```

#### POST /posts
Create a new post.

**Request Body:**
```json
{
  "pet_id": "uuid",
  "content": "Great walk in the park today!",
  "media_urls": ["https://example.com/walk.jpg"]
}
```

**Response:**
```json
{
  "id": "uuid",
  "author_user": {
    "id": "uuid",
    "name": "John Doe",
    "avatar_url": "https://example.com/john.jpg"
  },
  "pet": {
    "id": "uuid",
    "name": "Buddy",
    "breed": "Golden Retriever",
    "avatar_url": "https://example.com/buddy.jpg"
  },
  "content": "Great walk in the park today!",
  "media_urls": ["https://example.com/walk.jpg"],
  "created_at": "2023-01-01T00:00:00Z",
  "likes_count": 0,
  "comments_count": 0,
  "liked_by_user": false
}
```

#### GET /posts/:id
Get a specific post by ID.

**Response:**
```json
{
  "id": "uuid",
  "author_user": {
    "id": "uuid",
    "name": "John Doe",
    "avatar_url": "https://example.com/john.jpg"
  },
  "pet": {
    "id": "uuid",
    "name": "Buddy",
    "breed": "Golden Retriever",
    "avatar_url": "https://example.com/buddy.jpg"
  },
  "content": "Great walk in the park today!",
  "media_urls": ["https://example.com/walk.jpg"],
  "created_at": "2023-01-01T00:00:00Z",
  "likes_count": 5,
  "comments_count": 2,
  "liked_by_user": true,
  "comments": [
    {
      "id": "uuid",
      "author_user": {
        "id": "uuid",
        "name": "Jane Smith",
        "avatar_url": "https://example.com/jane.jpg"
      },
      "text": "Looks fun!",
      "created_at": "2023-01-01T01:00:00Z"
    }
  ]
}
```

#### POST /posts/:id/comments
Add a comment to a post.

**Request Body:**
```json
{
  "text": "Looks fun!"
}
```

**Response:**
```json
{
  "id": "uuid",
  "post_id": "uuid",
  "author_user": {
    "id": "uuid",
    "name": "John Doe",
    "avatar_url": "https://example.com/john.jpg"
  },
  "text": "Looks fun!",
  "created_at": "2023-01-01T01:00:00Z"
}
```

#### POST /posts/:id/likes
Like a post.

**Response:**
```json
{
  "id": "uuid",
  "post_id": "uuid",
  "user_id": "uuid",
  "created_at": "2023-01-01T01:00:00Z"
}
```

### Locations

#### GET /locations/nearby
Get nearby locations based on user's current location.

**Query Parameters:**
- `lat`: Latitude (required)
- `lng`: Longitude (required)
- `radius`: Search radius in kilometers (default: 5)
- `type`: Filter by location type (optional)

**Response:**
```json
{
  "locations": [
    {
      "id": "uuid",
      "name": "Central Park Dog Run",
      "type": "off_leash",
      "coordinates": {
        "lat": 40.7829,
        "lng": -73.9654
      },
      "amenities": ["water", "bags", "seating"],
      "rating": 4.5,
      "distance_km": 1.2
    }
  ]
}
```

#### GET /locations/:id
Get details for a specific location.

**Response:**
```json
{
  "id": "uuid",
  "name": "Central Park Dog Run",
  "type": "off_leash",
  "coordinates": {
    "lat": 40.7829,
    "lng": -73.9654
  },
  "amenities": ["water", "bags", "seating"],
  "rating": 4.5,
  "description": "A large, fenced-off area for dogs to run and play off-leash.",
  "hours": "6:00 AM - 9:00 PM",
  "photos": ["https://example.com/photo1.jpg"],
  "reviews": [
    {
      "user": {
        "name": "John Doe",
        "avatar_url": "https://example.com/john.jpg"
      },
      "rating": 5,
      "comment": "Great place for dogs to play!",
      "created_at": "2023-01-01T00:00:00Z"
    }
  ]
}
```

### Meetups

#### GET /meetups/suggest
Get suggested meetups based on location and compatibility.

**Query Parameters:**
- `lat`: Latitude (required)
- `lng`: Longitude (required)
- `pet_id`: Pet ID to find compatible matches for (required)

**Response:**
```json
{
  "suggestions": [
    {
      "pet": {
        "id": "uuid",
        "name": "Max",
        "breed": "Labrador",
        "age": 4,
        "avatar_url": "https://example.com/max.jpg"
      },
      "owner": {
        "id": "uuid",
        "name": "Jane Smith",
        "avatar_url": "https://example.com/jane.jpg"
      },
      "compatibility_score": 0.92,
      "distance_km": 0.8,
      "suggested_location": {
        "id": "uuid",
        "name": "Central Park Dog Run",
        "type": "off_leash",
        "coordinates": {
          "lat": 40.7829,
          "lng": -73.9654
        }
      },
      "estimated_arrival_time": "15 minutes"
    }
  ]
}
```

#### POST /meetups/:id/rsvp
RSVP to a meetup.

**Request Body:**
```json
{
  "status": "confirmed"
}
```

**Response:**
```json
{
  "id": "uuid",
  "meetup_id": "uuid",
  "user_id": "uuid",
  "pet_id": "uuid",
  "status": "confirmed",
  "created_at": "2023-01-01T00:00:00Z"
}
```

### Groups

#### GET /groups
Get groups the user belongs to.

**Response:**
```json
{
  "groups": [
    {
      "id": "uuid",
      "name": "NYC Golden Retriever Owners",
      "description": "A group for Golden Retriever owners in NYC",
      "member_count": 150,
      "is_admin": false,
      "joined_at": "2023-01-01T00:00:00Z"
    }
  ]
}
```

#### POST /groups
Create a new group.

**Request Body:**
```json
{
  "name": "NYC Golden Retriever Owners",
  "description": "A group for Golden Retriever owners in NYC",
  "is_private": false
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "NYC Golden Retriever Owners",
  "description": "A group for Golden Retriever owners in NYC",
  "is_private": false,
  "member_count": 1,
  "created_at": "2023-01-01T00:00:00Z"
}
```

#### GET /groups/:id/events
Get events for a specific group.

**Response:**
```json
{
  "events": [
    {
      "id": "uuid",
      "name": "Monthly Meetup",
      "description": "Monthly meetup for Golden Retriever owners",
      "location": {
        "id": "uuid",
        "name": "Central Park Dog Run",
        "coordinates": {
          "lat": 40.7829,
          "lng": -73.9654
        }
      },
      "scheduled_time": "2023-02-01T14:00:00Z",
      "attendee_count": 25,
      "is_attending": true
    }
  ]
}
```

#### POST /groups/:id/events
Create a new group event.

**Request Body:**
```json
{
  "name": "Monthly Meetup",
  "description": "Monthly meetup for Golden Retriever owners",
  "location_id": "uuid",
  "scheduled_time": "2023-02-01T14:00:00Z"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "Monthly Meetup",
  "description": "Monthly meetup for Golden Retriever owners",
  "location": {
    "id": "uuid",
    "name": "Central Park Dog Run",
    "coordinates": {
      "lat": 40.7829,
      "lng": -73.9654
    }
  },
  "scheduled_time": "2023-02-01T14:00:00Z",
  "attendee_count": 1,
  "created_at": "2023-01-01T00:00:00Z"
}
```

## Error Responses

All endpoints return appropriate HTTP status codes and error messages:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": {
      "field": "email",
      "message": "Invalid email format"
    }
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required or invalid token
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

## Rate Limiting

API endpoints are rate limited:
- **Authenticated requests**: 1000 requests per hour
- **Unauthenticated requests**: 100 requests per hour

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit per hour
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)

## Webhooks

The API supports webhooks for real-time notifications:

### POST /webhooks
Register a webhook endpoint.

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["meetup.suggested", "goal.achieved", "friend.request"]
}
```

**Response:**
```json
{
  "id": "uuid",
  "url": "https://your-app.com/webhook",
  "events": ["meetup.suggested", "goal.achieved", "friend.request"],
  "secret": "webhook_secret",
  "created_at": "2023-01-01T00:00:00Z"
}
```

## ML Service Endpoints

The ML service runs on port 8001 and provides the following endpoints:

### POST /predict/compatibility
Predict compatibility between two pets.

**Request Body:**
```json
{
  "pet_a_breed": "Golden Retriever",
  "pet_b_breed": "Labrador",
  "pet_a_age": 3,
  "pet_b_age": 4,
  "pet_a_size": "large",
  "pet_b_size": "large",
  "past_interactions": 5,
  "play_success_rate": 0.8,
  "energy_level_diff": 0.2
}
```

**Response:**
```json
{
  "compatibility_score": 0.85,
  "confidence": 0.9,
  "recommendation": "Good match. These pets should get along well."
}
```

### POST /predict/energy
Predict pet energy state.

**Request Body:**
```json
{
  "activity_duration": 120,
  "distance_walked": 5.2,
  "heart_rate_avg": 120,
  "rest_periods": 3,
  "play_time": 60,
  "social_interactions": 5,
  "time_outdoors": 180
}
```

**Response:**
```json
{
  "energy_state": "medium",
  "readiness_score": 0.75,
  "recommendations": [
    "Pet has moderate energy levels.",
    "Suitable for normal activities and light play.",
    "Good time for training sessions."
  ]
}
```