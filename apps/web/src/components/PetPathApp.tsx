'use client';

import React, { useState } from 'react';
import { Home, Activity, Trophy, MessageCircle, User } from 'lucide-react';
import { FeedScreen } from './FeedScreen';
import { ActivityScreen } from './ActivityScreen';
import { LeaderboardScreen } from './LeaderboardScreen';
import { MessagesScreen } from './MessagesScreen';
import { ProfileScreen } from './ProfileScreen';
import { ErrorBoundary } from './ErrorBoundary';

function PetPathApp() {
  const [activeTab, setActiveTab] = useState('feed');

  const renderScreen = () => {
    switch (activeTab) {
      case 'feed':
        return <FeedScreen />;
      case 'activity':
        return <ActivityScreen />;
      case 'leaderboard':
        return <LeaderboardScreen />;
      case 'messages':
        return <MessagesScreen />;
      case 'profile':
        return <ProfileScreen />;
      default:
        return <FeedScreen />;
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      {/* Mobile-first container */}
      <div className="w-full max-w-md h-screen bg-background flex flex-col relative">
        {/* Main Content */}
        <div className="flex-1 overflow-hidden">
          {renderScreen()}
        </div>

        {/* Bottom Navigation - Glass Morphism */}
        <nav className="glass-card border-t border-border/20 px-4 py-3 mx-4 mb-4 rounded-xl">
          <div className="flex justify-around items-center max-w-md mx-auto">
            <button
            onClick={() => setActiveTab('feed')}
            className={`relative p-3 rounded-xl transition-all duration-300 ${
              activeTab === 'feed'
                ? 'bg-accent text-white shadow-lg scale-110'
                : 'text-muted-foreground hover:text-foreground hover:scale-105'
            }`}
          >
            <Home size={24} />
          </button>
          <button
            onClick={() => setActiveTab('activity')}
            className={`relative p-3 rounded-xl transition-all duration-300 ${
              activeTab === 'activity'
                ? 'bg-accent text-white shadow-lg scale-110'
                : 'text-muted-foreground hover:text-foreground hover:scale-105'
            }`}
          >
            <Activity size={24} />
          </button>
          <button
            onClick={() => setActiveTab('leaderboard')}
            className={`relative p-3 rounded-xl transition-all duration-300 ${
              activeTab === 'leaderboard'
                ? 'bg-accent text-white shadow-lg scale-110'
                : 'text-muted-foreground hover:text-foreground hover:scale-105'
            }`}
          >
            <Trophy size={24} />
          </button>
          <button
            onClick={() => setActiveTab('messages')}
            className={`relative p-3 rounded-xl transition-all duration-300 ${
              activeTab === 'messages'
                ? 'bg-accent text-white shadow-lg scale-110'
                : 'text-muted-foreground hover:text-foreground hover:scale-105'
            }`}
          >
            <MessageCircle size={24} />
            {/* Notification dot */}
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-destructive rounded-full ring-2 ring-background animate-pulse"></div>
          </button>
          <button
            onClick={() => setActiveTab('profile')}
            className={`relative p-3 rounded-xl transition-all duration-300 ${
              activeTab === 'profile'
                ? 'bg-accent text-white shadow-lg scale-110'
                : 'text-muted-foreground hover:text-foreground hover:scale-105'
            }`}
          >
            <User size={24} />
          </button>
        </div>
      </nav>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <PetPathApp />
    </ErrorBoundary>
  );
}
