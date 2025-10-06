'use client';

import React, { useState } from 'react';
import {
  Home,
  Activity,
  Trophy,
  MessageCircle,
  User,
  Heart,
  Calendar,
  ShoppingBag,
  Shield,
  BarChart3
} from 'lucide-react';
import { FeedScreen } from './FeedScreen';
import { ActivityScreen } from './ActivityScreen';
import { LeaderboardScreen } from './LeaderboardScreen';
import { MessagesScreen } from './MessagesScreen';
import { ProfileScreen } from './ProfileScreen';
import { LoginScreen } from './auth/LoginScreen';
import { ErrorBoundary } from './ErrorBoundary';
import { useSessionStore } from '@/store/session';

// Beta MVP Screens
import { MatchDiscoveryScreen } from './matches/MatchDiscoveryScreen';
import { EventsScreen } from './events/EventsScreen';
import { ServicesHubScreen } from './services/ServicesHubScreen';
import { MeetupProposalScreen } from './meetups/MeetupProposalScreen';
import { EnhancedChatScreen } from './chat/EnhancedChatScreen';
import { VerificationScreen } from './verification/VerificationScreen';
import { AnalyticsDashboard } from './analytics/AnalyticsDashboard';

type TabType =
  | 'discover'
  | 'events'
  | 'services'
  | 'messages'
  | 'profile'
  | 'meetups'
  | 'verification'
  | 'analytics'
  | 'feed'
  | 'activity'
  | 'leaderboard';

function BetaApp() {
  const [activeTab, setActiveTab] = useState<TabType>('discover');
  const { isAuthenticated, user } = useSessionStore();

  // Show login screen if not authenticated
  if (!isAuthenticated) {
    return <LoginScreen />;
  }

  const renderScreen = () => {
    switch (activeTab) {
      case 'discover':
        return <MatchDiscoveryScreen />;
      case 'events':
        return <EventsScreen />;
      case 'services':
        return <ServicesHubScreen />;
      case 'messages':
        return <EnhancedChatScreen />;
      case 'meetups':
        return <MeetupProposalScreen />;
      case 'verification':
        return <VerificationScreen />;
      case 'analytics':
        return <AnalyticsDashboard />;
      case 'profile':
        return <ProfileScreen />;
      case 'feed':
        return <FeedScreen />;
      case 'activity':
        return <ActivityScreen />;
      case 'leaderboard':
        return <LeaderboardScreen />;
      default:
        return <MatchDiscoveryScreen />;
    }
  };

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* Top Navigation Bar */}
      <div className="border-b border-border/20 bg-background/95 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="text-2xl">🐾</div>
              <h1 className="text-xl font-bold">PetPath</h1>
              <span className="text-xs bg-accent text-white px-2 py-0.5 rounded-full">BETA</span>
            </div>

            {/* Secondary Nav */}
            <div className="hidden md:flex items-center gap-2">
              <NavButton
                icon={Calendar}
                label="Meetups"
                active={activeTab === 'meetups'}
                onClick={() => setActiveTab('meetups')}
              />
              <NavButton
                icon={Shield}
                label="Verify"
                active={activeTab === 'verification'}
                onClick={() => setActiveTab('verification')}
                badge={!user?.isVerified}
              />
              {user?.isAdmin && (
                <NavButton
                  icon={BarChart3}
                  label="Analytics"
                  active={activeTab === 'analytics'}
                  onClick={() => setActiveTab('analytics')}
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        {renderScreen()}
      </div>

      {/* Bottom Navigation - Glass Morphism */}
      <nav className="glass-card border-t border-border/20 px-4 py-3">
        <div className="flex justify-around items-center max-w-2xl mx-auto">
          <NavIconButton
            icon={Heart}
            active={activeTab === 'discover'}
            onClick={() => setActiveTab('discover')}
            label="Discover"
          />
          <NavIconButton
            icon={Calendar}
            active={activeTab === 'events'}
            onClick={() => setActiveTab('events')}
            label="Events"
          />
          <NavIconButton
            icon={ShoppingBag}
            active={activeTab === 'services'}
            onClick={() => setActiveTab('services')}
            label="Services"
          />
          <NavIconButton
            icon={MessageCircle}
            active={activeTab === 'messages'}
            onClick={() => setActiveTab('messages')}
            label="Messages"
            badge
          />
          <NavIconButton
            icon={User}
            active={activeTab === 'profile'}
            onClick={() => setActiveTab('profile')}
            label="Profile"
          />
        </div>
      </nav>
    </div>
  );
}

interface NavButtonProps {
  icon: any;
  label: string;
  active: boolean;
  onClick: () => void;
  badge?: boolean;
}

function NavButton({ icon: Icon, label, active, onClick, badge }: NavButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`relative px-4 py-2 rounded-lg transition-all duration-200 text-sm font-medium ${
        active
          ? 'bg-accent text-white shadow-md'
          : 'text-muted-foreground hover:text-foreground hover:bg-muted'
      }`}
    >
      <div className="flex items-center gap-2">
        <Icon className="w-4 h-4" />
        <span>{label}</span>
      </div>
      {badge && (
        <div className="absolute -top-1 -right-1 w-2 h-2 bg-accent rounded-full"></div>
      )}
    </button>
  );
}

interface NavIconButtonProps {
  icon: any;
  active: boolean;
  onClick: () => void;
  label: string;
  badge?: boolean;
}

function NavIconButton({ icon: Icon, active, onClick, label, badge }: NavIconButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`relative flex flex-col items-center gap-1 p-2 rounded-xl transition-all duration-300 ${
        active
          ? 'text-accent scale-110'
          : 'text-muted-foreground hover:text-foreground hover:scale-105'
      }`}
    >
      <Icon className="w-6 h-6" />
      <span className="text-xs font-medium">{label}</span>
      {badge && (
        <div className="absolute top-1 right-1 w-2 h-2 bg-destructive rounded-full ring-2 ring-background"></div>
      )}
    </button>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <BetaApp />
    </ErrorBoundary>
  );
}
