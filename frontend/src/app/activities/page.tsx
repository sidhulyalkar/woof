'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { 
  Plus, 
  Play, 
  Pause, 
  Stop, 
  MapPin, 
  Clock, 
  Flame, 
  Footprints, 
  TrendingUp, 
  Target,
  Calendar,
  Award,
  Zap
} from 'lucide-react'

export default function ActivitiesPage() {
  const [isActivityOpen, setIsActivityOpen] = useState(false)
  const [isTracking, setIsTracking] = useState(false)
  const [currentActivity, setCurrentActivity] = useState(null)

  // Mock data for activities
  const recentActivities = [
    {
      id: '1',
      type: 'Walk',
      pet: 'Buddy',
      duration: 45,
      distance: 3.2,
      calories: 180,
      steps: 4250,
      location: 'Central Park',
      startTime: '2024-01-15T08:00:00Z',
      endTime: '2024-01-15T08:45:00Z',
      weather: { temp: 18, condition: 'Sunny' },
      mood: 'Happy'
    },
    {
      id: '2',
      type: 'Play',
      pet: 'Buddy',
      duration: 30,
      distance: 0.5,
      calories: 120,
      steps: 1200,
      location: 'Dog Park',
      startTime: '2024-01-14T17:00:00Z',
      endTime: '2024-01-14T17:30:00Z',
      weather: { temp: 22, condition: 'Partly Cloudy' },
      mood: 'Energetic'
    },
    {
      id: '3',
      type: 'Walk',
      pet: 'Whiskers',
      duration: 20,
      distance: 0.8,
      calories: 60,
      steps: 800,
      location: 'Neighborhood',
      startTime: '2024-01-14T19:00:00Z',
      endTime: '2024-01-14T19:20:00Z',
      weather: { temp: 16, condition: 'Clear' },
      mood: 'Calm'
    }
  ]

  const weeklyStats = {
    totalActivities: 12,
    totalDistance: 24.6,
    totalCalories: 1450,
    totalSteps: 32450,
    avgDuration: 35,
    mostActivePet: 'Buddy',
    favoriteLocation: 'Central Park'
  }

  const monthlyGoals = {
    steps: { current: 84320, target: 100000, unit: 'steps' },
    distance: { current: 68.4, target: 80, unit: 'km' },
    calories: { current: 4200, target: 5000, unit: 'cal' },
    activities: { current: 18, target: 25, unit: 'activities' }
  }

  const achievements = [
    { title: 'Early Bird', description: '5 morning walks before 7 AM', icon: '🌅', unlocked: true },
    { title: 'Weekend Warrior', description: '10 weekend activities', icon: '🏆', unlocked: true },
    { title: 'Marathon Walker', description: '100km in a month', icon: '🏃', unlocked: false },
    { title: 'Social Butterfly', description: '15 playdates with friends', icon: '🦋', unlocked: false }
  ]

  const startActivity = (type, pet) => {
    setIsTracking(true)
    setCurrentActivity({
      type,
      pet,
      startTime: new Date(),
      duration: 0,
      distance: 0,
      steps: 0
    })
  }

  const stopActivity = () => {
    setIsTracking(false)
    setIsActivityOpen(true)
  }

  const ActivityCard = ({ activity }) => (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
              activity.type === 'Walk' ? 'bg-blue-100' : 
              activity.type === 'Play' ? 'bg-green-100' : 
              'bg-purple-100'
            }`}>
              {activity.type === 'Walk' ? <Footprints className="w-5 h-5 text-blue-600" /> :
               activity.type === 'Play' ? <Play className="w-5 h-5 text-green-600" /> :
               <Zap className="w-5 h-5 text-purple-600" />}
            </div>
            <div>
              <CardTitle className="text-lg">{activity.type}</CardTitle>
              <CardDescription>with {activity.pet} • {activity.location}</CardDescription>
            </div>
          </div>
          <Badge variant={activity.mood === 'Happy' ? 'default' : activity.mood === 'Energetic' ? 'secondary' : 'outline'}>
            {activity.mood}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <Clock className="w-4 h-4 mx-auto mb-1 text-gray-600" />
            <p className="font-semibold">{activity.duration} min</p>
            <p className="text-xs text-gray-600">Duration</p>
          </div>
          <div className="text-center">
            <MapPin className="w-4 h-4 mx-auto mb-1 text-gray-600" />
            <p className="font-semibold">{activity.distance} km</p>
            <p className="text-xs text-gray-600">Distance</p>
          </div>
          <div className="text-center">
            <Flame className="w-4 h-4 mx-auto mb-1 text-gray-600" />
            <p className="font-semibold">{activity.calories}</p>
            <p className="text-xs text-gray-600">Calories</p>
          </div>
          <div className="text-center">
            <Footprints className="w-4 h-4 mx-auto mb-1 text-gray-600" />
            <p className="font-semibold">{activity.steps.toLocaleString()}</p>
            <p className="text-xs text-gray-600">Steps</p>
          </div>
        </div>
        <div className="mt-3 flex items-center justify-between text-sm text-gray-600">
          <span>{new Date(activity.startTime).toLocaleDateString()}</span>
          <span>{activity.weather.temp}°C {activity.weather.condition}</span>
        </div>
      </CardContent>
    </Card>
  )

  const GoalProgress = ({ goal, icon: Icon }) => (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Icon className="w-5 h-5 text-gray-600" />
            <span className="font-medium capitalize">{goal.unit}</span>
          </div>
          <span className="text-sm text-gray-600">
            {goal.current.toLocaleString()} / {goal.target.toLocaleString()}
          </span>
        </div>
        <Progress value={(goal.current / goal.target) * 100} className="h-2" />
        <p className="text-xs text-gray-600 mt-1">
          {Math.round((goal.current / goal.target) * 100)}% complete
        </p>
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
              <span className="text-xl font-bold text-gray-900">Activities</span>
            </div>
            <div className="flex items-center space-x-4">
              {isTracking ? (
                <Button onClick={stopActivity} variant="destructive" size="sm">
                  <Stop className="w-4 h-4 mr-2" />
                  Stop Activity
                </Button>
              ) : (
                <Dialog open={isActivityOpen} onOpenChange={setIsActivityOpen}>
                  <DialogTrigger asChild>
                    <Button>
                      <Plus className="w-4 h-4 mr-2" />
                      Log Activity
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-md">
                    <DialogHeader>
                      <DialogTitle>Log Activity</DialogTitle>
                      <DialogDescription>Record your pet's activity</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="activity-type">Activity Type</Label>
                        <Select>
                          <SelectTrigger>
                            <SelectValue placeholder="Select activity type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="walk">Walk</SelectItem>
                            <SelectItem value="play">Play</SelectItem>
                            <SelectItem value="training">Training</SelectItem>
                            <SelectItem value="hike">Hike</SelectItem>
                            <SelectItem value="swim">Swim</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label htmlFor="pet-select">Pet</Label>
                        <Select>
                          <SelectTrigger>
                            <SelectValue placeholder="Select pet" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="buddy">Buddy</SelectItem>
                            <SelectItem value="whiskers">Whiskers</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="duration">Duration (minutes)</Label>
                          <Input id="duration" type="number" placeholder="30" />
                        </div>
                        <div>
                          <Label htmlFor="distance">Distance (km)</Label>
                          <Input id="distance" type="number" step="0.1" placeholder="2.5" />
                        </div>
                      </div>
                      <div>
                        <Label htmlFor="location">Location</Label>
                        <Input id="location" placeholder="e.g., Central Park" />
                      </div>
                      <div>
                        <Label htmlFor="notes">Notes (optional)</Label>
                        <Textarea id="notes" placeholder="How was the activity?" />
                      </div>
                      <Button className="w-full">Log Activity</Button>
                    </div>
                  </DialogContent>
                </Dialog>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Activity Tracker */}
      {isTracking && currentActivity && (
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-4">
          <div className="max-w-7xl mx-auto">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center">
                  <Play className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-lg font-semibold">Active {currentActivity.type}</p>
                  <p className="text-blue-100">with {currentActivity.pet}</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold">
                  {Math.floor((Date.now() - currentActivity.startTime) / 60000)} min
                </p>
                <p className="text-blue-100">Duration</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="activities">Activities</TabsTrigger>
            <TabsTrigger value="goals">Goals</TabsTrigger>
            <TabsTrigger value="achievements">Achievements</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="space-y-6">
            {/* Weekly Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <Card>
                <CardContent className="p-4 text-center">
                  <Calendar className="w-8 h-8 mx-auto mb-2 text-blue-600" />
                  <p className="text-2xl font-bold">{weeklyStats.totalActivities}</p>
                  <p className="text-sm text-gray-600">Activities</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <MapPin className="w-8 h-8 mx-auto mb-2 text-green-600" />
                  <p className="text-2xl font-bold">{weeklyStats.totalDistance} km</p>
                  <p className="text-sm text-gray-600">Distance</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <Flame className="w-8 h-8 mx-auto mb-2 text-red-600" />
                  <p className="text-2xl font-bold">{weeklyStats.totalCalories}</p>
                  <p className="text-sm text-gray-600">Calories</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <Footprints className="w-8 h-8 mx-auto mb-2 text-purple-600" />
                  <p className="text-2xl font-bold">{weeklyStats.totalSteps.toLocaleString()}</p>
                  <p className="text-sm text-gray-600">Steps</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <Clock className="w-8 h-8 mx-auto mb-2 text-orange-600" />
                  <p className="text-2xl font-bold">{weeklyStats.avgDuration} min</p>
                  <p className="text-sm text-gray-600">Avg Duration</p>
                </CardContent>
              </Card>
            </div>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
                <CardDescription>Start a new activity with your pet</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Button 
                    onClick={() => startActivity('Walk', 'Buddy')}
                    className="h-20 flex-col"
                    variant="outline"
                  >
                    <Footprints className="w-6 h-6 mb-2" />
                    Walk with Buddy
                  </Button>
                  <Button 
                    onClick={() => startActivity('Play', 'Buddy')}
                    className="h-20 flex-col"
                    variant="outline"
                  >
                    <Play className="w-6 h-6 mb-2" />
                    Play with Buddy
                  </Button>
                  <Button 
                    onClick={() => startActivity('Walk', 'Whiskers')}
                    className="h-20 flex-col"
                    variant="outline"
                  >
                    <Footprints className="w-6 h-6 mb-2" />
                    Walk with Whiskers
                  </Button>
                  <Button 
                    onClick={() => startActivity('Play', 'Whiskers')}
                    className="h-20 flex-col"
                    variant="outline"
                  >
                    <Play className="w-6 h-6 mb-2" />
                    Play with Whiskers
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Recent Activities */}
            <Card>
              <CardHeader>
                <CardTitle>Recent Activities</CardTitle>
                <CardDescription>Your latest pet activities</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {recentActivities.slice(0, 3).map((activity) => (
                    <ActivityCard key={activity.id} activity={activity} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="activities" className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">All Activities</h2>
              <Select defaultValue="all">
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Activities</SelectItem>
                  <SelectItem value="walk">Walks</SelectItem>
                  <SelectItem value="play">Play Sessions</SelectItem>
                  <SelectItem value="training">Training</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-4">
              {recentActivities.map((activity) => (
                <ActivityCard key={activity.id} activity={activity} />
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="goals" className="space-y-6">
            <h2 className="text-2xl font-bold">Monthly Goals</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <GoalProgress goal={monthlyGoals.steps} icon={Footprints} />
              <GoalProgress goal={monthlyGoals.distance} icon={MapPin} />
              <GoalProgress goal={monthlyGoals.calories} icon={Flame} />
              <GoalProgress goal={monthlyGoals.activities} icon={Calendar} />
            </div>
            
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  Goal Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <p className="font-medium text-blue-900">Increase Daily Steps</p>
                    <p className="text-sm text-blue-700">Try to reach 8,000 steps per day with Buddy</p>
                  </div>
                  <div className="p-3 bg-green-50 rounded-lg">
                    <p className="font-medium text-green-900">Weekend Adventures</p>
                    <p className="text-sm text-green-700">Plan longer weekend activities to boost distance</p>
                  </div>
                  <div className="p-3 bg-purple-50 rounded-lg">
                    <p className="font-medium text-purple-900">Social Activities</p>
                    <p className="text-sm text-purple-700">Join group activities to increase social time</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="achievements" className="space-y-6">
            <h2 className="text-2xl font-bold">Achievements</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {achievements.map((achievement, index) => (
                <Card key={index} className={achievement.unlocked ? 'border-yellow-200 bg-yellow-50' : 'opacity-60'}>
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-3">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl ${
                        achievement.unlocked ? 'bg-yellow-400' : 'bg-gray-300'
                      }`}>
                        {achievement.icon}
                      </div>
                      <div>
                        <p className="font-semibold">{achievement.title}</p>
                        <p className="text-sm text-gray-600">{achievement.description}</p>
                        {achievement.unlocked && (
                          <Badge variant="secondary" className="mt-1">Unlocked</Badge>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
            
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="w-5 h-5" />
                  Progress to Next Achievement
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="font-medium">Marathon Walker</span>
                      <span className="text-sm text-gray-600">68.4/100 km</span>
                    </div>
                    <Progress value={68.4} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="font-medium">Social Butterfly</span>
                      <span className="text-sm text-gray-600">8/15 playdates</span>
                    </div>
                    <Progress value={(8/15)*100} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}