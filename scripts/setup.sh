#!/bin/bash

# Woof Development Setup Script
# Run this to set up your development environment

set -e

echo "🐾 Woof Development Setup"
echo "========================="
echo ""

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null; then
    echo "❌ pnpm not found. Installing pnpm..."
    npm install -g pnpm
    echo "✅ pnpm installed"
else
    echo "✅ pnpm is installed"
fi

# Check Node version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js 18+ required. Current version: $(node -v)"
    echo "   Please install Node.js 20+ from https://nodejs.org"
    exit 1
else
    echo "✅ Node.js version: $(node -v)"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pnpm install

# Set up database package
echo ""
echo "💾 Setting up database..."
cd packages/database

if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please update packages/database/.env with your PostgreSQL credentials"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update packages/database/.env with your database credentials"
echo "2. Run 'pnpm db:generate' to generate Prisma client"
echo "3. Run 'pnpm db:migrate' to create database tables"
echo "4. Run 'pnpm db:seed' to populate with demo data"
echo "5. Run 'pnpm dev' to start all services"
echo ""
echo "🐾 Happy coding!"
