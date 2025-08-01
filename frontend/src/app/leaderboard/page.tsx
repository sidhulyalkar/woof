'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { 
  Trophy, 
  MapPin, 
  TrendingUp, 
  Calendar, 
  Users, 
  Award,
  Target,
  Footprints,
  Flame,
  Star,
  Crown,
  Medal,
  Zap
} from 'lucide-react'

export default function LeaderboardPage() {
  const [timeFilter, setTimeFilter] = useState('week')
  const [locationFilter, setLocationFilter] = useState('global')
  const [categoryFilter, setCategoryFilter] = useState('steps')

  // Mock data for leaderboards
  const globalLeaderboard = [
    { 
      rank: 1, 
      name: 'Sarah Johnson', 
      petName: 'Max', 
      petBreed: 'Labrador', 
      avatar: '/api/placeholder/48/48', 
      steps: 45230, 
      distance: 89.4, 
      calories: 3240, 
      activities: 18,
      points: 4850,
      change: '+2',
      isCurrentUser: false 
    },
    { 
      rank: 2, 
      name: 'Mike Chen', 
      petName: 'Luna', 
      petBreed: 'Husky', 
      avatar: '/api/placeholder/48/48', 
      steps: 42180, 
      distance: 82.1, 
      calories: 2980, 
      activities: 16,
      points: 4620,
      change: '-1',
      isCurrentUser: false 
    },
    { 
      rank: 3, 
      name: 'Emma Davis', 
      petName: 'Charlie', 
      petBreed: 'Beagle', 
      avatar: '/api/placeholder/48/48', 
      steps: 38920, 
      distance: 76.8, 
      calories: 2750, 
      activities: 15,
      points: 4280,
      change: '+1',
      isCurrentUser: false 
    },
    { 
      rank: 4, 
      name: 'Alex Rodriguez', 
      petName: 'Bella', 
      petBreed: 'Poodle', 
      avatar: '/api/placeholder/48/48', 
      steps: 35640, 
      distance: 70.2, 
      calories: 2510, 
      activities: 14,
      points: 3950,
      change: '+3',
      isCurrentUser: false 
    },
    { 
      rank: 5, 
      name: 'Lisa Wang', 
      petName: 'Mochi', 
      petBreed: 'Shiba Inu', 
      avatar: '/api/placeholder/48/48', 
      steps: 33890, 
      distance: 66.7, 
      calories: 2380, 
      activities: 13,
      points: 3720,
      change: '-2',
      isCurrentUser: false 
    },
    { 
      rank: 7, 
      name: 'You', 
      petName: 'Buddy', 
      petBreed: 'Golden Retriever', 
      avatar: '/api/placeholder/48/48', 
      steps: 28450, 
      distance: 56.8, 
      calories: 1980, 
      activities: 12,
      points: 3120,
      change: '+4',
      isCurrentUser: true 
    }
  ]

  const localLeaderboard = [
    { 
      rank: 1, 
      name: 'Sarah Johnson', 
      petName: 'Max', 
      petBreed: 'Labrador', 
      avatar: '/api/placeholder/48/48', 
      steps: 45230, 
      distance: 89.4, 
      calories: 3240, 
      activities: 18,
      points: 4850,
      distance: 0.8,
      isCurrentUser: false 
    },
    { 
      rank: 2, 
      name: 'You', 
      petName: 'Buddy', 
      petBreed: 'Golden Retriever', 
      avatar: '/api/placeholder/48/48', 
      steps: 28450, 
      distance: 56.8, 
      calories: 1980, 
      activities: 12,
      points: 3120,
      distance: 1.2,
      isCurrentUser: true 
    },
    { 
      rank: 3, 
      name: 'David Kim', 
      petName: 'Cookie', 
      petBreed: 'Corgi', 
      avatar: '/api/placeholder/48/48', 
      steps: 26780, 
      distance: 52.3, 
      calories: 1850, 
      activities: 11,
      points: 2940,
      distance: 2.1,
      isCurrentUser: false 
    }
  ]

  const breedLeaderboard = [
    { 
      rank: 1, 
      name: 'Sarah Johnson', 
      petName: 'Max', 
      petBreed: 'Labrador', 
      avatar: '/api/placeholder/48/48', 
      steps: 45230, 
      distance: 89.4, 
      calories: 3240, 
      activities: 18,
      points: 4850,
      isCurrentUser: false 
    },
    { 
      rank: 2, 
      name: 'You', 
      petName: 'Buddy', 
      petBreed: 'Golden Retriever', 
      avatar: '/api/placeholder/48/48', 
      steps: 28450, 
      distance: 56.8, 
      calories: 1980, 
      activities: 12,
      points: 3120,
      isCurrentUser: true 
    },
    { 
      rank: 3, 
      name: 'Tom Wilson', 
      petName: 'Rocky', 
      petBreed: 'Golden Retriever', 
      avatar: '/api/placeholder/48/48', 
      steps: 27340, 
      distance: 54.2, 
      calories: 1890, 
      activities: 11,
      points: 3010,
      isCurrentUser: false 
    }
  ]

  const achievements = [
    { 
      title: 'Marathon Walker', 
      description: 'Walk 100km in a month', 
      icon: '🏃', 
      progress: 68.4, 
      target: 100, 
      unit: 'km',
      unlocked: false 
    },
    { 
      title: 'Social Butterfly', 
      description: 'Make 15 pet friends', 
      icon: '🦋', 
      progress: 8, 
      target: 15, 
      unit: 'friends',
      unlocked: false 
    },
    { 
      title: 'Early Bird', 
      description: 'Complete 10 morning walks before 7 AM', 
      icon: '🌅', 
      progress: 7, 
      target: 10, 
      unit: 'walks',
      unlocked: false 
    },
    { 
      title: 'Weekend Warrior', 
      description: 'Complete 20 weekend activities', 
      icon: '🏆', 
      progress: 18, 
      target: 20, 
      unit: 'activities',
      unlocked: false 
    },
    { 
      title: 'Step Master', 
      description: 'Reach 50,000 steps in a week', 
      icon: '👟', 
      progress: 42380, 
      target: 50000, 
      unit: 'steps',
      unlocked: false 
    }
  ]

  const weeklyChallenges = [
    { 
      title: 'Step Challenge', 
      description: 'Reach 10,000 steps daily', 
      progress: 68420, 
      target: 70000, 
      participants: 1240,
      reward: 500,
      icon: Footprints 
    },
    { 
      title: 'Distance Challenge', 
      description: 'Walk 25km this week', 
      progress: 18.4, 
      target: 25, 
      participants: 890,
      reward: 300,
      icon: MapPin 
    },
    { 
      title: 'Social Challenge', 
      description: 'Arrange 3 playdates', 
      progress: 1, 
      target: 3, 
      participants: 456,
      reward: 400,
      icon: Users 
    }
  ]

  const LeaderboardRow = ({ entry, showDistance = false }) => (
    <div className={`flex items-center justify-between p-4 rounded-lg transition-colors ${
      entry.isCurrentUser 
        ? 'bg-blue-50 border-2 border-blue-200' 
        : 'hover:bg-gray-50'
    }`}>
      <div className="flex items-center space-x-4">
        {/* Rank Badge */}
        <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-lg ${
          entry.rank === 1 ? 'bg-gradient-to-br from-yellow-400 to-yellow-600' :
          entry.rank === 2 ? 'bg-gradient-to-br from-gray-300 to-gray-500' :
          entry.rank === 3 ? 'bg-gradient-to-br from-orange-400 to-orange-600' :
          'bg-gray-400'
        }`}>
          {entry.rank === 1 ? <Crown className="w-5 h-5" /> : 
           entry.rank === 2 ? <Medal className="w-5 h-5" /> : 
           entry.rank === 3 ? <Award className="w-5 h-5" /> : 
           entry.rank}
        </div>

        {/* User Info */}
        <Avatar className="w-12 h-12">
          <AvatarImage src={entry.avatar} />
          <AvatarFallback>{entry.name.charAt(0)}</AvatarFallback>
        </Avatar>

        <div>
          <div className="flex items-center space-x-2">
            <p className={`font-medium ${entry.isCurrentUser ? 'text-blue-600' : ''}`}>
              {entry.name}
              {entry.isCurrentUser && <Badge variant="secondary" className="ml-2">You</Badge>}
            </p>
            {entry.change && (
              <span className={`text-sm font-medium ${
                entry.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
              }`}>
                {entry.change}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-600">
            {entry.petName} • {entry.petBreed}
            {showDistance && entry.distance && (
              <span className="ml-2 text-xs text-gray-500">
                {entry.distance}km away
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Stats */}
      <div className="text-right">
        <div className="flex items-center space-x-4">
          <div className="text-right">
            <p className="font-bold text-lg">{entry.steps.toLocaleString()}</p>
            <p className="text-xs text-gray-600">steps</p>
          </div>
          <div className="text-right">
            <p className="font-bold text-lg">{entry.points.toLocaleString()}</p>
            <p className="text-xs text-gray-600">points</p>
          </div>
        </div>
      </div>
    </div>
  )

  const AchievementCard = ({ achievement }) => (
    <Card className={`h-full ${achievement.unlocked ? 'border-yellow-200 bg-yellow-50' : ''}`}>
      <CardContent className="p-4">
        <div className="flex items-center space-x-3 mb-3">
          <div className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl ${
            achievement.unlocked ? 'bg-yellow-400' : 'bg-gray-300'
          }`}>
            {achievement.icon}
          </div>
          <div>
            <p className="font-medium">{achievement.title}</p>
            <p className="text-sm text-gray-600">{achievement.description}</p>
          </div>
        </div>
        <div className="mt-3">
          <div className="flex justify-between text-sm mb-1">
            <span>Progress</span>
            <span>
              {typeof achievement.progress === 'number' 
                ? achievement.progress.toLocaleString() 
                : achievement.progress} / {achievement.target} {achievement.unit}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ 
                width: `${(achievement.progress / achievement.target) * 100}%` 
              }}
            />
          </div>
          <p className="text-xs text-gray-600 mt-1">
            {Math.round((achievement.progress / achievement.target) * 100)}% complete
          </p>
        </div>
      </CardContent>
    </Card>
  )

  const ChallengeCard = ({ challenge }) => (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
              <challenge.icon className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="font-medium">{challenge.title}</p>
              <p className="text-sm text-gray-600">{challenge.description}</p>
            </div>
          </div>
          <Badge variant="outline">
            {challenge.participants} participants
          </Badge>
        </div>
        <div className="mb-3">
          <div className="flex justify-between text-sm mb-1">
            <span>Progress</span>
            <span>
              {challenge.progress} / {challenge.target} {typeof challenge.progress === 'number' && challenge.progress < 10 ? 'km' : ''}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-green-600 h-2 rounded-full transition-all duration-300"
              style={{ 
                width: `${(challenge.progress / challenge.target) * 100}%` 
              }}
            />
          </div>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-1">
            <Star className="w-4 h-4 text-yellow-500" />
            <span className="text-sm font-medium">{challenge.reward} pts</span>
          </div>
          <Button size="sm">Join Challenge</Button>
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <span className="text-xl font-bold text-gray-900">Leaderboard</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Select value={timeFilter} onValueChange={setTimeFilter}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="day">Today</SelectItem>
                    <SelectItem value="week">This Week</SelectItem>
                    <SelectItem value="month">This Month</SelectItem>
                    <SelectItem value="all">All Time</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="steps">Steps</SelectItem>
                    <SelectItem value="distance">Distance</SelectItem>
                    <SelectItem value="calories">Calories</SelectItem>
                    <SelectItem value="points">Points</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs defaultValue="global" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="global">Global</TabsTrigger>
            <TabsTrigger value="local">Local</TabsTrigger>
            <TabsTrigger value="breed">Breed</TabsTrigger>
            <TabsTrigger value="challenges">Challenges</TabsTrigger>
          </TabsList>
          
          <TabsContent value="global" className="space-y-6">
            {/* Your Stats Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-4 text-center">
                  <Trophy className="w-8 h-8 mx-auto mb-2 text-yellow-500" />
                  <p className="text-2xl font-bold">#7</p>
                  <p className="text-sm text-gray-600">Global Rank</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <Footprints className="w-8 h-8 mx-auto mb-2 text-blue-600" />
                  <p className="text-2xl font-bold">28.4K</p>
                  <p className="text-sm text-gray-600">Steps</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <Flame className="w-8 h-8 mx-auto mb-2 text-red-600" />
                  <p className="text-2xl font-bold">1,980</p>
                  <p className="text-sm text-gray-600">Calories</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <Star className="w-8 h-8 mx-auto mb-2 text-purple-600" />
                  <p className="text-2xl font-bold">3,120</p>
                  <p className="text-sm text-gray-600">Points</p>
                </CardContent>
              </Card>
            </div>

            {/* Global Leaderboard */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="w-5 h-5" />
                  Global Leaderboard
                </CardTitle>
                <CardDescription>
                  Top pet owners worldwide this {timeFilter}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {globalLeaderboard.map((entry) => (
                    <LeaderboardRow key={entry.rank} entry={entry} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="local" className="space-y-6">
            {/* Location Selector */}
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">Local Leaderboard</h2>
                <p className="text-gray-600">Top pet owners in your area</p>
              </div>
              <Select value={locationFilter} onValueChange={setLocationFilter}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="global">Global</SelectItem>
                  <SelectItem value="local">New York City</SelectItem>
                  <SelectItem value="brooklyn">Brooklyn</SelectItem>
                  <SelectItem value="manhattan">Manhattan</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Local Leaderboard */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="w-5 h-5" />
                  New York City Rankings
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {localLeaderboard.map((entry) => (
                    <LeaderboardRow key={entry.rank} entry={entry} showDistance={true} />
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Nearby Top Performers */}
            <Card>
              <CardHeader>
                <CardTitle>Nearby Top Performers</CardTitle>
                <CardDescription>High achievers within 5km of you</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {localLeaderboard.slice(0, 2).map((entry) => (
                    <div key={entry.rank} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                        entry.rank === 1 ? 'bg-yellow-500' : 'bg-gray-400'
                      }`}>
                        {entry.rank}
                      </div>
                      <Avatar className="w-10 h-10">
                        <AvatarImage src={entry.avatar} />
                        <AvatarFallback>{entry.name.charAt(0)}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <p className="font-medium">{entry.name}</p>
                        <p className="text-sm text-gray-600">{entry.petName}</p>
                      </div>
                      <div className="text-right">
                        <p className="font-bold">{entry.steps.toLocaleString()}</p>
                        <p className="text-xs text-gray-600">{entry.distance}km</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="breed" className="space-y-6">
            {/* Breed Leaderboard */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  Golden Retriever Leaderboard
                </CardTitle>
                <CardDescription>Top Golden Retriever owners this week</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {breedLeaderboard.map((entry) => (
                    <LeaderboardRow key={entry.rank} entry={entry} />
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Breed Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardContent className="p-4 text-center">
                  <p className="text-3xl font-bold text-blue-600">247</p>
                  <p className="text-sm text-gray-600">Golden Retrievers</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <p className="text-3xl font-bold text-green-600">15.2K</p>
                  <p className="text-sm text-gray-600">Avg Steps/Day</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <p className="text-3xl font-bold text-purple-600">4.2</p>
                  <p className="text-sm text-gray-600">Avg Activities/Week</p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="challenges" className="space-y-6">
            {/* Active Challenges */}
            <div>
              <h2 className="text-2xl font-bold mb-4">Active Challenges</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {weeklyChallenges.map((challenge, index) => (
                  <ChallengeCard key={index} challenge={challenge} />
                ))}
              </div>
            </div>

            {/* Your Achievements */}
            <div>
              <h2 className="text-2xl font-bold mb-4">Your Achievements</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {achievements.map((achievement, index) => (
                  <AchievementCard key={index} achievement={achievement} />
                ))}
              </div>
            </div>

            {/* Achievement Progress */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  Achievement Progress
                </CardTitle>
                <CardDescription>Your progress towards unlocking achievements</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {achievements.slice(0, 3).map((achievement, index) => (
                    <div key={index}>
                      <div className="flex justify-between mb-2">
                        <span className="font-medium">{achievement.title}</span>
                        <span className="text-sm text-gray-600">
                          {achievement.progress} / {achievement.target} {achievement.unit}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300"
                          style={{ 
                            width: `${(achievement.progress / achievement.target) * 100}%` 
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}