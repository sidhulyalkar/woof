#!/bin/bash

# Woof API Test Script
# Tests all major endpoints

API_URL="http://localhost:4000/api/v1"

echo "🐾 Testing Woof API..."
echo ""

# 1. Health Check
echo "1️⃣ Health Check"
curl -s $API_URL/../health | jq '.'
echo ""

# 2. Register a new user
echo "2️⃣ Register New User"
REGISTER_RESPONSE=$(curl -s -X POST $API_URL/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "handle": "testuser",
    "email": "test@woof.com",
    "password": "password123",
    "bio": "Test user from script"
  }')
echo $REGISTER_RESPONSE | jq '.'
TOKEN=$(echo $REGISTER_RESPONSE | jq -r '.access_token')
echo "Token: $TOKEN"
echo ""

# 3. Login with demo user
echo "3️⃣ Login with Demo User"
LOGIN_RESPONSE=$(curl -s -X POST $API_URL/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@woof.com",
    "password": "password123"
  }')
echo $LOGIN_RESPONSE | jq '.'
DEMO_TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')
echo ""

# 4. Get current user
echo "4️⃣ Get Current User Profile"
curl -s $API_URL/auth/me \
  -H "Authorization: Bearer $DEMO_TOKEN" | jq '.'
echo ""

# 5. Get all pets
echo "5️⃣ Get All Pets"
curl -s $API_URL/pets \
  -H "Authorization: Bearer $DEMO_TOKEN" | jq '.'
echo ""

# 6. Get social feed
echo "6️⃣ Get Social Feed"
curl -s $API_URL/social/posts \
  -H "Authorization: Bearer $DEMO_TOKEN" | jq '.'
echo ""

echo "✅ All tests complete!"
echo ""
echo "📚 Open Swagger docs: http://localhost:4000/docs"
