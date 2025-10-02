-- Woof Database Initialization Script
-- This script sets up the PostgreSQL database with pgvector extension

-- Create vector extension for ML embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Create n8n database
CREATE DATABASE woof_n8n;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE woof TO postgres;
GRANT ALL PRIVILEGES ON DATABASE woof_n8n TO postgres;

-- Log initialization
\echo 'Woof database initialized successfully!'
\echo 'pgvector extension enabled for ML compatibility scoring'
