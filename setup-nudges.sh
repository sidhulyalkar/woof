#!/bin/bash

# PetPath Nudge Engine Setup Script
# This script sets up the proactive nudge system

set -e  # Exit on error

echo "🐾 PetPath Nudge Engine Setup"
echo "=============================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "⚠️  Docker is not running. Please start Docker Desktop first."
    echo "   Then run this script again."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Start database services
echo "📦 Starting database services..."
docker compose up -d postgres n8n

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker compose exec -T postgres pg_isready -U woof > /dev/null 2>&1; do
    echo "   Still waiting..."
    sleep 2
done

echo "✅ PostgreSQL is ready"
echo ""

# Run database migrations
echo "🔄 Running database migrations..."
cd packages/database
npm run db:migrate -- --name add_chat_and_nudge_improvements

# Generate Prisma client
echo "🔧 Generating Prisma client..."
npm run db:generate

echo "✅ Database setup complete"
echo ""

# Check for VAPID keys
cd ../..
if ! grep -q "VAPID_PUBLIC_KEY" apps/api/.env 2>/dev/null || [ -z "$(grep VAPID_PUBLIC_KEY apps/api/.env | cut -d'=' -f2)" ]; then
    echo "🔑 Generating VAPID keys for push notifications..."

    # Generate VAPID keys
    VAPID_OUTPUT=$(npx web-push generate-vapid-keys 2>/dev/null)
    PUBLIC_KEY=$(echo "$VAPID_OUTPUT" | grep "Public Key:" | cut -d':' -f2 | xargs)
    PRIVATE_KEY=$(echo "$VAPID_OUTPUT" | grep "Private Key:" | cut -d':' -f2 | xargs)

    # Add to .env if not exists
    if ! grep -q "VAPID_PUBLIC_KEY" apps/api/.env 2>/dev/null; then
        echo "" >> apps/api/.env
        echo "# Web Push VAPID Keys" >> apps/api/.env
        echo "VAPID_PUBLIC_KEY=$PUBLIC_KEY" >> apps/api/.env
        echo "VAPID_PRIVATE_KEY=$PRIVATE_KEY" >> apps/api/.env
    fi

    # Add to web .env
    if [ -f "apps/web/.env.local" ]; then
        if ! grep -q "NEXT_PUBLIC_VAPID_PUBLIC_KEY" apps/web/.env.local; then
            echo "" >> apps/web/.env.local
            echo "NEXT_PUBLIC_VAPID_PUBLIC_KEY=$PUBLIC_KEY" >> apps/web/.env.local
        fi
    fi

    echo "✅ VAPID keys generated and added to .env files"
else
    echo "✅ VAPID keys already configured"
fi
echo ""

# Rebuild API to pick up Prisma client changes
echo "🏗️  Rebuilding API with updated Prisma client..."
cd apps/api
npm run build

echo ""
echo "✨ Setup complete! Next steps:"
echo ""
echo "1. Review the setup documentation:"
echo "   cat NUDGE_ENGINE_SETUP.md"
echo ""
echo "2. Start the development servers:"
echo "   npm run dev  (from root)"
echo ""
echo "3. Test proximity nudges:"
echo "   curl -X POST http://localhost:3001/api/nudges/check/proximity"
echo ""
echo "4. Implement frontend push subscription (see NUDGE_ENGINE_SETUP.md)"
echo ""
echo "🎉 The Proactive Nudge Engine is ready!"
