'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Plus, Edit, Heart, Activity, MapPin, Calendar, TrendingUp } from 'lucide-react'

export default function PetsPage() {
  const [selectedPet, setSelectedPet] = useState(null)
  const [isAddPetOpen, setIsAddPetOpen] = useState(false)
  const [isEditPetOpen, setIsEditPetOpen] = useState(false)

  // Mock data for user's pets
  const userPets = [
    {
      id: '1',
      name: 'Buddy',
      breed: 'Golden Retriever',
      age: 3,
      size: 'large',
      weight: 32.5,
      bio: 'Friendly and energetic golden retriever who loves playing fetch and swimming.',
      avatar: '/api/placeholder/150/150',
      deviceId: 'airtag-buddy-123',
      energy: 85,
      happiness: 92,
      todayActivity: { steps: 8432, distance: 6.2, calories: 342 },
      weeklyStats: { walks: 12, distance: 68.4, calories: 2840 },
      achievements: ['Early Bird', 'Social Butterfly', 'Adventure Seeker'],
      friends: 8,
      photos: 24
    },
    {
      id: '2',
      name: 'Whiskers',
      breed: 'Siamese',
      age: 5,
      size: 'small',
      weight: 4.2,
      bio: 'Elegant Siamese cat who enjoys sunny spots and gentle play sessions.',
      avatar: '/api/placeholder/150/150',
      deviceId: 'airtag-whiskers-456',
      energy: 45,
      happiness: 78,
      todayActivity: { steps: 2103, distance: 1.8, calories: 98 },
      weeklyStats: { walks: 6, distance: 12.8, calories: 560 },
      achievements: ['Sun Lover', 'Quiet Companion'],
      friends: 3,
      photos: 18
    }
  ]

  const activityHistory = [
    { date: '2024-01-15', type: 'Walk', duration: 45, distance: 3.2, calories: 180, location: 'Central Park' },
    { date: '2024-01-14', type: 'Play', duration: 30, distance: 0.5, calories: 120, location: 'Dog Park' },
    { date: '2024-01-14', type: 'Walk', duration: 60, distance: 4.1, calories: 220, location: 'Riverside Park' },
    { date: '2024-01-13', type: 'Training', duration: 25, distance: 0.2, calories: 90, location: 'Home' },
  ]

  const PetCard = ({ pet }) => (
    <Card className="cursor-pointer hover:shadow-lg transition-all duration-200" onClick={() => setSelectedPet(pet)}>
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
  )

  const PetDetails = ({ pet }) => (
    <div className="space-y-6">
      {/* Pet Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Avatar className="w-20 h-20">
                <AvatarImage src={pet.avatar} />
                <AvatarFallback className="text-2xl">{pet.name.charAt(0)}</AvatarFallback>
              </Avatar>
              <div>
                <CardTitle className="text-2xl">{pet.name}</CardTitle>
                <CardDescription className="text-lg">{pet.breed} • {pet.age} years old</CardDescription>
                <div className="flex items-center space-x-2 mt-2">
                  <Badge variant="outline">{pet.size}</Badge>
                  <Badge variant="outline">{pet.weight} kg</Badge>
                  <Badge variant={pet.energy > 70 ? 'default' : pet.energy > 40 ? 'secondary' : 'destructive'}>
                    {pet.energy > 70 ? 'Energetic' : pet.energy > 40 ? 'Calm' : 'Resting'}
                  </Badge>
                </div>
              </div>
            </div>
            <Dialog open={isEditPetOpen} onOpenChange={setIsEditPetOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <Edit className="w-4 h-4 mr-2" />
                  Edit Profile
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-md">
                <DialogHeader>
                  <DialogTitle>Edit Pet Profile</DialogTitle>
                  <DialogDescription>Update your pet's information</DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="name">Name</Label>
                    <Input id="name" defaultValue={pet.name} />
                  </div>
                  <div>
                    <Label htmlFor="breed">Breed</Label>
                    <Input id="breed" defaultValue={pet.breed} />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="age">Age</Label>
                      <Input id="age" type="number" defaultValue={pet.age} />
                    </div>
                    <div>
                      <Label htmlFor="weight">Weight (kg)</Label>
                      <Input id="weight" type="number" step="0.1" defaultValue={pet.weight} />
                    </div>
                  </div>
                  <div>
                    <Label htmlFor="size">Size</Label>
                    <Select defaultValue={pet.size}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="small">Small</SelectItem>
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="large">Large</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="bio">Bio</Label>
                    <Textarea id="bio" defaultValue={pet.bio} />
                  </div>
                  <Button className="w-full">Save Changes</Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-gray-700">{pet.bio}</p>
        </CardContent>
      </Card>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <Activity className="w-8 h-8 mx-auto mb-2 text-blue-600" />
            <p className="text-2xl font-bold">{pet.weeklyStats.walks}</p>
            <p className="text-sm text-gray-600">Walks this week</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <MapPin className="w-8 h-8 mx-auto mb-2 text-green-600" />
            <p className="text-2xl font-bold">{pet.weeklyStats.distance} km</p>
            <p className="text-sm text-gray-600">Distance</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <Heart className="w-8 h-8 mx-auto mb-2 text-red-600" />
            <p className="text-2xl font-bold">{pet.friends}</p>
            <p className="text-sm text-gray-600">Friends</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <Calendar className="w-8 h-8 mx-auto mb-2 text-purple-600" />
            <p className="text-2xl font-bold">{pet.photos}</p>
            <p className="text-sm text-gray-600">Photos</p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs for detailed information */}
      <Tabs defaultValue="activity" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="activity">Activity</TabsTrigger>
          <TabsTrigger value="achievements">Achievements</TabsTrigger>
          <TabsTrigger value="friends">Friends</TabsTrigger>
        </TabsList>
        
        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Recent Activity
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {activityHistory.map((activity, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                        <Activity className="w-5 h-5 text-blue-600" />
                      </div>
                      <div>
                        <p className="font-medium">{activity.type}</p>
                        <p className="text-sm text-gray-600">{activity.location}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">{activity.duration} min</p>
                      <p className="text-sm text-gray-600">{activity.distance} km</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="achievements" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Achievements</CardTitle>
              <CardDescription>Badges and milestones earned</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {pet.achievements.map((achievement, index) => (
                  <div key={index} className="text-center p-4 bg-gradient-to-br from-yellow-50 to-orange-50 rounded-lg border border-yellow-200">
                    <div className="w-12 h-12 bg-yellow-400 rounded-full flex items-center justify-center mx-auto mb-2">
                      <Trophy className="w-6 h-6 text-white" />
                    </div>
                    <p className="font-medium text-sm">{achievement}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="friends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Pet Friends</CardTitle>
              <CardDescription>{pet.name}'s social connections</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Array.from({ length: pet.friends }, (_, i) => (
                  <div key={i} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                    <Avatar className="w-12 h-12">
                      <AvatarImage src={`/api/placeholder/48/48`} />
                      <AvatarFallback>F{i + 1}</AvatarFallback>
                    </Avatar>
                    <div>
                      <p className="font-medium">Friend {i + 1}</p>
                      <p className="text-sm text-gray-600">Golden Retriever</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )

  const AddPetForm = () => (
    <div className="space-y-4">
      <div>
        <Label htmlFor="pet-name">Pet Name</Label>
        <Input id="pet-name" placeholder="Enter your pet's name" />
      </div>
      <div>
        <Label htmlFor="pet-breed">Breed</Label>
        <Input id="pet-breed" placeholder="e.g., Golden Retriever, Siamese" />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="pet-age">Age</Label>
          <Input id="pet-age" type="number" placeholder="Age in years" />
        </div>
        <div>
          <Label htmlFor="pet-weight">Weight (kg)</Label>
          <Input id="pet-weight" type="number" step="0.1" placeholder="Weight in kg" />
        </div>
      </div>
      <div>
        <Label htmlFor="pet-size">Size</Label>
        <Select>
          <SelectTrigger>
            <SelectValue placeholder="Select size" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="small">Small</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="large">Large</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div>
        <Label htmlFor="pet-device">Device ID (optional)</Label>
        <Input id="pet-device" placeholder="AirTag or GPS device ID" />
      </div>
      <div>
        <Label htmlFor="pet-bio">Bio</Label>
        <Textarea id="pet-bio" placeholder="Tell us about your pet's personality" />
      </div>
      <Button className="w-full">Add Pet</Button>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <button onClick={() => setSelectedPet(null)} className="flex items-center">
                <span className="text-xl font-bold text-gray-900">PetPath</span>
              </button>
              {selectedPet && (
                <span className="ml-4 text-gray-600">/ {selectedPet.name}</span>
              )}
            </div>
            <div className="flex items-center space-x-4">
              <Dialog open={isAddPetOpen} onOpenChange={setIsAddPetOpen}>
                <DialogTrigger asChild>
                  <Button>
                    <Plus className="w-4 h-4 mr-2" />
                    Add Pet
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-md">
                  <DialogHeader>
                    <DialogTitle>Add New Pet</DialogTitle>
                    <DialogDescription>Add a new pet to your profile</DialogDescription>
                  </DialogHeader>
                  <AddPetForm />
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!selectedPet ? (
          <div>
            <div className="mb-6">
              <h1 className="text-2xl font-bold text-gray-900">My Pets</h1>
              <p className="text-gray-600">Manage your pets and track their activities</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {userPets.map((pet) => (
                <PetCard key={pet.id} pet={pet} />
              ))}
            </div>
          </div>
        ) : (
          <PetDetails pet={selectedPet} />
        )}
      </main>
    </div>
  )
}