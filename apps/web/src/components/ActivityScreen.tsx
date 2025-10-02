import React from 'react';
import { Clock, MapPin, Zap, Target, Calendar, Play, Footprints, Heart, Trophy } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

const mockActivities = [
  {
    id: 1,
    pet: { name: 'Buddy', avatar: 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=150&h=150&fit=crop&crop=face' },
    type: 'Walk',
    duration: '32 min',
    distance: '2.4 km',
    calories: 180,
    location: 'Central Park',
    time: '2 hours ago'
  },
  {
    id: 2,
    pet: { name: 'Luna', avatar: 'https://images.unsplash.com/photo-1551717743-49959800b1f6?w=150&h=150&fit=crop&crop=face' },
    type: 'Run',
    duration: '45 min',
    distance: '5.2 km',
    calories: 320,
    location: 'Riverside Trail',
    time: '4 hours ago'
  },
  {
    id: 3,
    pet: { name: 'Whiskers', avatar: 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=150&h=150&fit=crop&crop=face' },
    type: 'Play',
    duration: '25 min',
    distance: '0.8 km',
    calories: 95,
    location: 'Home',
    time: '6 hours ago'
  }
];

const dailyGoals = {
  steps: { current: 8420, target: 10000 },
  calories: { current: 285, target: 400 },
  distance: { current: 3.2, target: 5.0 },
  playTime: { current: 95, target: 120 }
};

export function ActivityScreen() {
  return (
    <div className="bg-background min-h-screen">
      {/* Header */}
      <div className="sticky top-0 z-50 glass-card border-b border-border/20 px-4 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Activity</h1>
          <Button className="rounded-full bg-accent hover:bg-accent/90">
            <Play size={20} />
            Start Workout
          </Button>
        </div>
      </div>

      <div className="p-4 pb-24 space-y-6">
        {/* Quick Stats Ring */}
        <div className="glass-card p-6 rounded-xl">
          <div className="grid grid-cols-2 gap-6">
            <div className="text-center">
              <div className="relative w-24 h-24 mx-auto mb-2">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle
                    cx="48"
                    cy="48"
                    r="40"
                    stroke="rgba(107, 168, 255, 0.2)"
                    strokeWidth="8"
                    fill="none"
                  />
                  <circle
                    cx="48"
                    cy="48"
                    r="40"
                    stroke="#6BA8FF"
                    strokeWidth="8"
                    fill="none"
                    strokeDasharray={`${(dailyGoals.steps.current / dailyGoals.steps.target) * 251.2} 251.2`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Footprints size={20} className="text-accent" />
                </div>
              </div>
              <p className="text-lg font-semibold">{dailyGoals.steps.current.toLocaleString()}</p>
              <p className="text-sm text-muted-foreground">Steps</p>
            </div>
            <div className="text-center">
              <div className="relative w-24 h-24 mx-auto mb-2">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle
                    cx="48"
                    cy="48"
                    r="40"
                    stroke="rgba(37, 193, 138, 0.2)"
                    strokeWidth="8"
                    fill="none"
                  />
                  <circle
                    cx="48"
                    cy="48"
                    r="40"
                    stroke="#25C18A"
                    strokeWidth="8"
                    fill="none"
                    strokeDasharray={`${(dailyGoals.calories.current / dailyGoals.calories.target) * 251.2} 251.2`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Zap size={20} className="text-success" />
                </div>
              </div>
              <p className="text-lg font-semibold">{dailyGoals.calories.current}</p>
              <p className="text-sm text-muted-foreground">Calories</p>
            </div>
          </div>
        </div>

        {/* Daily Goals */}
        <div className="glass-card p-6 rounded-xl">
          <div className="flex items-center gap-2 mb-4">
            <Target size={20} className="text-accent" />
            <h2 className="text-lg font-semibold">Today's Goals</h2>
          </div>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">Distance</span>
                <span className="text-sm text-accent font-semibold">{dailyGoals.distance.current}km / {dailyGoals.distance.target}km</span>
              </div>
              <Progress 
                value={(dailyGoals.distance.current / dailyGoals.distance.target) * 100} 
                className="h-3" 
              />
            </div>
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">Play Time</span>
                <span className="text-sm text-accent font-semibold">{dailyGoals.playTime.current}min / {dailyGoals.playTime.target}min</span>
              </div>
              <Progress 
                value={(dailyGoals.playTime.current / dailyGoals.playTime.target) * 100} 
                className="h-3" 
              />
            </div>
          </div>
        </div>

        {/* Weekly Stats */}
        <div className="glass-card p-6 rounded-xl">
          <div className="flex items-center gap-2 mb-4">
            <Calendar size={20} className="text-accent" />
            <h2 className="text-lg font-semibold">This Week</h2>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-4 bg-surface-elevated/50 rounded-lg">
              <div className="w-10 h-10 bg-accent/20 rounded-full flex items-center justify-center mx-auto mb-2">
                <MapPin size={20} className="text-accent" />
              </div>
              <p className="text-xl font-semibold text-accent">12.4km</p>
              <p className="text-sm text-muted-foreground">Distance</p>
            </div>
            <div className="text-center p-4 bg-surface-elevated/50 rounded-lg">
              <div className="w-10 h-10 bg-success/20 rounded-full flex items-center justify-center mx-auto mb-2">
                <Zap size={20} className="text-success" />
              </div>
              <p className="text-xl font-semibold text-success">1,240</p>
              <p className="text-sm text-muted-foreground">Calories</p>
            </div>
            <div className="text-center p-4 bg-surface-elevated/50 rounded-lg">
              <div className="w-10 h-10 bg-warning/20 rounded-full flex items-center justify-center mx-auto mb-2">
                <Clock size={20} className="text-warning" />
              </div>
              <p className="text-xl font-semibold text-warning">7h 32m</p>
              <p className="text-sm text-muted-foreground">Active</p>
            </div>
          </div>
        </div>

        {/* Recent Activities */}
        <div>
          <h2 className="text-lg font-semibold mb-4 px-2">Recent Activities</h2>
          <div className="space-y-3">
            {mockActivities.map((activity) => (
              <div key={activity.id} className="glass-card p-4 rounded-xl">
                <div className="flex items-center gap-3 mb-4">
                  <Avatar className="w-12 h-12 ring-2 ring-accent/20">
                    <AvatarImage src={activity.pet.avatar} />
                    <AvatarFallback className="bg-accent/20">{activity.pet.name[0]}</AvatarFallback>
                  </Avatar>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold">{activity.pet.name}</span>
                      <Badge 
                        variant="secondary" 
                        className={`text-xs ${
                          activity.type === 'Run' ? 'bg-success/20 text-success border-success/30' : 
                          activity.type === 'Walk' ? 'bg-accent/20 text-accent border-accent/30' :
                          'bg-warning/20 text-warning border-warning/30'
                        }`}
                      >
                        {activity.type}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">{activity.time}</p>
                  </div>
                  <Trophy size={18} className="text-accent" />
                </div>
                
                <div className="grid grid-cols-4 gap-3">
                  <div className="text-center p-2 bg-surface-elevated/30 rounded-lg">
                    <Clock size={16} className="text-muted-foreground mx-auto mb-1" />
                    <p className="text-sm font-medium">{activity.duration}</p>
                  </div>
                  <div className="text-center p-2 bg-surface-elevated/30 rounded-lg">
                    <MapPin size={16} className="text-muted-foreground mx-auto mb-1" />
                    <p className="text-sm font-medium">{activity.distance}</p>
                  </div>
                  <div className="text-center p-2 bg-surface-elevated/30 rounded-lg">
                    <Zap size={16} className="text-muted-foreground mx-auto mb-1" />
                    <p className="text-sm font-medium">{activity.calories}</p>
                  </div>
                  <div className="text-center p-2 bg-surface-elevated/30 rounded-lg">
                    <Heart size={16} className="text-muted-foreground mx-auto mb-1" />
                    <p className="text-xs font-medium truncate">{activity.location}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}