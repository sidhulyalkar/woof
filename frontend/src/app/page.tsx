'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Heart, Footprints, Trophy, MapPin, Users, TrendingUp, Sun, Camera, MessageCircle, Share2 } from 'lucide-react'

export default function Home() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [happinessScore, setHappinessScore] = useState(87)
  const [weeklyProgress, setWeeklyProgress] = useState(68)

  // Mock data for demonstration
  const userPets = [
    {
      id: '1',
      name: 'Buddy',
      breed: 'Golden Retriever',
      age: 3,
      avatar: '/api/placeholder/100/100',
      energy: 85,
      happiness: 92,
      todayActivity: { steps: 8432, distance: 6.2, calories: 342 },
    },
    {
      id: '2',
      name: 'Whiskers',
      breed: 'Siamese',
      age: 5,
      avatar: '/api/placeholder/100/100',
      energy: 45,
      happiness: 78,
      todayActivity: { steps: 2103, distance: 1.8, calories: 98 },
    }
  ]

  const leaderboardData = [
    { rank: 1, name: 'Sarah Johnson', petName: 'Max', steps: 45230, avatar: '/api/placeholder/40/40' },
    { rank: 2, name: 'Mike Chen', petName: 'Luna', steps: 42180, avatar: '/api/placeholder/40/40' },
    { rank: 3, name: 'Emma Davis', petName: 'Charlie', steps: 38920, avatar: '/api/placeholder/40/40' },
    { rank: 7, name: 'You', petName: 'Buddy', steps: 28450, avatar: '/api/placeholder/40/40', isCurrentUser: true },
  ]

  const socialFeed = [
    {
      id: '1',
      user: { name: 'Sarah Johnson', avatar: '/api/placeholder/40/40' },
      pet: { name: 'Max', breed: 'Labrador' },
      content: 'Amazing morning hike with Max at Central Park! The fall colors are incredible 🍁',
      image: '/api/placeholder/400/300',
      likes: 24,
      comments: 8,
      time: '2 hours ago',
      location: 'Central Park, NYC'
    },
    {
      id: '2',
      user: { name: 'Mike Chen', avatar: '/api/placeholder/40/40' },
      pet: { name: 'Luna', breed: 'Husky' },
      content: 'Luna made a new friend at the dog park today! They played for hours 🐕',
      image: '/api/placeholder/400/300',
      likes: 31,
      comments: 12,
      time: '4 hours ago',
      location: 'Brooklyn Dog Park'
    }
  ]

  const happinessMetrics = [
    { icon: Footprints, label: 'Walks', value: 12, target: 14, unit: 'this week', color: 'bg-blue-500' },
    { icon: Sun, label: 'Outdoor Time', value: 18, target: 20, unit: 'hours', color: 'bg-yellow-500' },
    { icon: Users, label: 'Social Time', value: 6, target: 8, unit: 'sessions', color: 'bg-green-500' },
    { icon: Camera, label: 'Photos Shared', value: 8, target: 10, unit: 'this week', color: 'bg-purple-500' },
  ]

  const Dashboard = () => (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-6 text-white">
        <h1 className="text-3xl font-bold mb-2">Welcome back! 👋</h1>
        <p className="text-blue-100">Ready for another adventure with your furry friends?</p>
        
        {/* Happiness Score */}
        <div className="mt-6 bg-white/20 rounded-xl p-4 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-100">Weekly Happiness Score</p>
              <p className="text-3xl font-bold">{happinessScore}%</p>
            </div>
            <div className="relative w-20 h-20">
              <svg className="w-full h-full transform -rotate-90">
                <circle cx="40" cy="40" r="36" stroke="rgba(255,255,255,0.3)" strokeWidth="8" fill="none" />
                <circle cx="40" cy="40" r="36" stroke="white" strokeWidth="8" fill="none"
                  strokeDasharray={`${(happinessScore / 100) * 226} 226`} strokeLinecap="round" />
              </svg>
              <Heart className="absolute inset-0 m-auto w-8 h-8 text-white" />
            </div>
          </div>
        </div>
      </div>

      {/* Pet Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {userPets.map((pet) => (
          <Card key={pet.id} className="overflow-hidden hover:shadow-lg transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Avatar className="w-12 h-12">
                    <AvatarImage src={pet.avatar} />
                    <AvatarFallback>{pet.name.charAt(0)}</AvatarFallback>
                  </Avatar>
                  <div>
                    <CardTitle className="text-lg">{pet.name}</CardTitle>
                    <CardDescription>{pet.breed} • {pet.age} years</CardDescription>
                  </div>
                </div>
                <Badge variant={pet.energy > 70 ? 'default' : pet.energy > 40 ? 'secondary' : 'destructive'}>
                  {pet.energy > 70 ? 'Energetic' : pet.energy > 40 ? 'Calm' : 'Resting'}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Today's Activity</span>
                  <span className="font-medium">{pet.todayActivity.steps.toLocaleString()} steps</span>
                </div>
                <Progress value={(pet.todayActivity.steps / 10000) * 100} className="h-2" />
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="bg-gray-50 rounded-lg p-2">
                    <p className="text-xs text-gray-600">Distance</p>
                    <p className="font-semibold">{pet.todayActivity.distance} km</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-2">
                    <p className="text-xs text-gray-600">Calories</p>
                    <p className="font-semibold">{pet.todayActivity.calories}</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-2">
                    <p className="text-xs text-gray-600">Happiness</p>
                    <p className="font-semibold">{pet.happiness}%</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Weekly Progress */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Weekly Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Monthly Goal</span>
              <span className="text-sm text-gray-600">{weeklyProgress}% Complete</span>
            </div>
            <Progress value={weeklyProgress} className="h-3" />
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-blue-50 rounded-lg">
                <p className="text-2xl font-bold text-blue-600">68.4</p>
                <p className="text-sm text-gray-600">km walked</p>
              </div>
              <div className="text-center p-3 bg-green-50 rounded-lg">
                <p className="text-2xl font-bold text-green-600">12</p>
                <p className="text-sm text-gray-600">activities</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )

  const Leaderboard = () => (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Trophy className="w-5 h-5" />
            Global Leaderboard
          </CardTitle>
          <CardDescription>Top pet owners this week</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {leaderboardData.map((entry) => (
              <div key={entry.rank} className={`flex items-center justify-between p-3 rounded-lg ${
                entry.isCurrentUser ? 'bg-blue-50 border border-blue-200' : 'hover:bg-gray-50'
              }`}>
                <div className="flex items-center space-x-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                    entry.rank === 1 ? 'bg-yellow-500' : entry.rank === 2 ? 'bg-gray-400' : entry.rank === 3 ? 'bg-orange-600' : 'bg-gray-300 text-gray-700'
                  }`}>
                    {entry.rank}
                  </div>
                  <Avatar className="w-10 h-10">
                    <AvatarImage src={entry.avatar} />
                    <AvatarFallback>{entry.name.charAt(0)}</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-medium">{entry.name}</p>
                    <p className="text-sm text-gray-600">{entry.petName}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-bold">{entry.steps.toLocaleString()}</p>
                  <p className="text-sm text-gray-600">steps</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MapPin className="w-5 h-5" />
            Local Leaderboard
          </CardTitle>
          <CardDescription>Top pet owners in your area</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {leaderboardData.slice(0, 5).map((entry) => (
              <div key={entry.rank} className="flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                    entry.rank === 1 ? 'bg-yellow-500' : entry.rank === 2 ? 'bg-gray-400' : entry.rank === 3 ? 'bg-orange-600' : 'bg-gray-300 text-gray-700'
                  }`}>
                    {entry.rank}
                  </div>
                  <Avatar className="w-10 h-10">
                    <AvatarImage src={entry.avatar} />
                    <AvatarFallback>{entry.name.charAt(0)}</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-medium">{entry.name}</p>
                    <p className="text-sm text-gray-600">{entry.petName}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-bold">{Math.floor(entry.steps * 0.8).toLocaleString()}</p>
                  <p className="text-sm text-gray-600">steps</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )

  const Social = () => (
    <div className="space-y-6">
      {/* Happiness Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Heart className="w-5 h-5 text-red-500" />
            Your Happiness Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            {happinessMetrics.map((metric, index) => (
              <div key={index} className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <metric.icon className={`w-6 h-6 ${metric.color.replace('bg-', 'text-')}`} />
                  <span className="text-sm text-gray-600">{metric.value}/{metric.target}</span>
                </div>
                <p className="font-medium mb-2">{metric.label}</p>
                <Progress value={(metric.value / metric.target) * 100} className="h-2" />
                <p className="text-xs text-gray-600 mt-1">{metric.unit}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Social Feed */}
      <div className="space-y-4">
        <h2 className="text-xl font-bold">Social Feed</h2>
        {socialFeed.map((post) => (
          <Card key={post.id} className="overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Avatar className="w-10 h-10">
                    <AvatarImage src={post.user.avatar} />
                    <AvatarFallback>{post.user.name.charAt(0)}</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-medium">{post.user.name}</p>
                    <p className="text-sm text-gray-600">with {post.pet.name} • {post.time}</p>
                  </div>
                </div>
                <Button variant="ghost" size="sm">
                  <MapPin className="w-4 h-4 mr-1" />
                  {post.location}
                </Button>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="mb-4">{post.content}</p>
              <div className="rounded-lg overflow-hidden mb-4">
                <div className="w-full h-48 bg-gradient-to-r from-blue-400 to-purple-500 flex items-center justify-center">
                  <Camera className="w-12 h-12 text-white" />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex space-x-4">
                  <Button variant="ghost" size="sm" className="text-gray-600">
                    <Heart className="w-4 h-4 mr-1" />
                    {post.likes}
                  </Button>
                  <Button variant="ghost" size="sm" className="text-gray-600">
                    <MessageCircle className="w-4 h-4 mr-1" />
                    {post.comments}
                  </Button>
                  <Button variant="ghost" size="sm" className="text-gray-600">
                    <Share2 className="w-4 h-4 mr-1" />
                    Share
                  </Button>
                </div>
                <Badge variant="secondary">
                  {post.pet.breed}
                </Badge>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Footprints className="w-8 h-8 text-blue-600 mr-2" />
              <span className="text-xl font-bold text-gray-900">PetPath</span>
            </div>
            <nav className="hidden md:flex space-x-8">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'dashboard' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('leaderboard')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'leaderboard' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Leaderboard
              </button>
              <button
                onClick={() => setActiveTab('social')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'social' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Social
              </button>
            </nav>
            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="sm">
                <Trophy className="w-4 h-4 mr-1" />
                2,450 pts
              </Button>
              <Avatar className="w-8 h-8">
                <AvatarImage src="/api/placeholder/32/32" />
                <AvatarFallback>JD</AvatarFallback>
              </Avatar>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'leaderboard' && <Leaderboard />}
        {activeTab === 'social' && <Social />}
      </main>
    </div>
  )
}