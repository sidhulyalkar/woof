"""
PetPath Backend API
FastAPI application for the PetPath social fitness platform for pets and owners
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
from typing import List, Optional

# Database imports will be added when models are defined
# from .database import engine, get_db
# from .models import Base
# from .routers import auth, users, pets, social, activities, meetups

security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("Starting PetPath Backend API...")
    # Base.metadata.create_all(bind=engine)  # Uncomment when models are defined
    yield
    # Shutdown
    print("Shutting down PetPath Backend API...")


app = FastAPI(
    title="PetPath API",
    description="Social fitness platform API for pets and their owners",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PetPath API is running"}


# Sample pets endpoint for testing
@app.get("/pets", tags=["Pets"])
async def get_pets():
    """Get all pets (sample endpoint)"""
    return [
        {
            "id": "1",
            "name": "Buddy",
            "breed": "Golden Retriever",
            "age": 3,
            "owner_id": "user-1",
            "avatar_url": "https://example.com/buddy.jpg"
        },
        {
            "id": "2", 
            "name": "Whiskers",
            "breed": "Siamese",
            "age": 5,
            "owner_id": "user-2",
            "avatar_url": "https://example.com/whiskers.jpg"
        }
    ]


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to PetPath API",
        "docs": "/docs",
        "version": "1.0.0"
    }


# Include routers (will be uncommented when routers are created)
# app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
# app.include_router(users.router, prefix="/users", tags=["Users"])
# app.include_router(pets.router, prefix="/pets", tags=["Pets"])
# app.include_router(social.router, prefix="/social", tags=["Social"])
# app.include_router(activities.router, prefix="/activities", tags=["Activities"])
# app.include_router(meetups.router, prefix="/meetups", tags=["Meetups"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )