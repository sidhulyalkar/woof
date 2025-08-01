'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { 
  Heart, 
  MessageCircle, 
  Share2, 
  Plus, 
  Camera, 
  MapPin, 
  Users, 
  TrendingUp,
  Paw,
  Sun,
  Smile,
  Filter,
  Search
} from 'lucide-react'

export default function SocialPage() {
  const [isCreatePostOpen, setIsCreatePostOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('feed')
  const [likedPosts, setLikedPosts] = useState(new Set())
  const [searchQuery, setSearchQuery] = useState('')

  // Mock data for social feed
  const socialFeed = [
    {
      id: '1',
      user: { 
        name: 'Sarah Johnson', 
        username: '@sarahj', 
        avatar: '/api/placeholder/48/48',
        isFollowing: true 
      },
      pet: { 
        name: 'Max', 
        breed: 'Labrador', 
        age: 4,
        avatar: '/api/placeholder/40/40'
      },
      content: 'Amazing morning hike with Max at Central Park! The fall colors are incredible 🍁 We met so many friendly dogs and Max made a new friend named Luna. Perfect way to start the day!',
      image: '/api/placeholder/600/400',
      likes: 24,
      comments: 8,
      shares: 3,
      time: '2 hours ago',
      location: 'Central Park, NYC',
      weather: { temp: 18, condition: 'Sunny' },
      activity: { type: 'Hike', duration: 45, distance: 3.2 },
      tags: ['#MorningHike', '#CentralPark', '#LabradorLife'],
      mood: 'Excited'
    },
    {
      id: '2',
      user: { 
        name: 'Mike Chen', 
        username: '@mikechen', 
        avatar: '/api/placeholder/48/48',
        isFollowing: true 
      },
      pet: { 
        name: 'Luna', 
        breed: 'Husky', 
        age: 3,
        avatar: '/api/placeholder/40/40'
      },
      content: 'Luna made a new friend at the dog park today! They played for hours 🐕 She\'s so happy and tired now. Nothing better than seeing your pup having fun with friends!',
      image: '/api/placeholder/600/400',
      likes: 31,
      comments: 12,
      shares: 2,
      time: '4 hours ago',
      location: 'Brooklyn Dog Park',
      weather: { temp: 22, condition: 'Partly Cloudy' },
      activity: { type: 'Play', duration: 90, distance: 0.8 },
      tags: ['#DogPark', '#PlayTime', '#Husky'],
      mood: 'Happy'
    },
    {
      id: '3',
      user: { 
        name: 'Emma Davis', 
        username: '@emmadavis', 
        avatar: '/api/placeholder/48/48',
        isFollowing: false 
      },
      pet: { 
        name: 'Charlie', 
        breed: 'Beagle', 
        age: 2,
        avatar: '/api/placeholder/40/40'
      },
      content: 'First time taking Charlie to the beach! He was a bit scared at first but ended up loving the water 🌊 We found some great shells and had a wonderful walk along the shore.',
      image: '/api/placeholder/600/400',
      likes: 18,
      comments: 6,
      shares: 1,
      time: '6 hours ago',
      location: 'Rockaway Beach',
      weather: { temp: 20, condition: 'Sunny' },
      activity: { type: 'Walk', duration: 60, distance: 2.1 },
      tags: ['#BeachDay', '#FirstTime', '#BeagleAdventures'],
      mood: 'Proud'
    }
  ]

  const friendsActivity = [
    {
      id: '1',
      user: { name: 'Alex Rodriguez', avatar: '/api/placeholder/32/32' },
      pet: { name: 'Bella', breed: 'Poodle' },
      activity: 'Completed a 5km walk',
      time: '30 minutes ago',
      location: 'Prospect Park'
    },
    {
      id: '2',
      user: { name: 'Lisa Wang', avatar: '/api/placeholder/32/32' },
      pet: { name: 'Mochi', breed: 'Shiba Inu' },
      activity: 'Reached 10,000 steps goal',
      time: '1 hour ago',
      location: 'Neighborhood'
    },
    {
      id: '3',
      user: { name: 'David Kim', avatar: '/api/placeholder/32/32' },
      pet: { name: 'Cookie', breed: 'Corgi' },
      activity: 'Made a new friend at the park',
      time: '2 hours ago',
      location: 'Dog Run'
    }
  ]

  const trendingTopics = [
    { topic: '#MorningWalks', posts: 1240, change: '+15%' },
    { topic: '#DogParkLife', posts: 890, change: '+8%' },
    { topic: '#PuppyTraining', posts: 654, change: '+22%' },
    { topic: '#BeachDay', posts: 432, change: '+5%' },
    { topic: '#HikingAdventures', posts: 321, change: '+18%' }
  ]

  const handleLike = (postId) => {
    const newLikedPosts = new Set(likedPosts)
    if (likedPosts.has(postId)) {
      newLikedPosts.delete(postId)
    } else {
      newLikedPosts.add(postId)
    }
    setLikedPosts(newLikedPosts)
  }

  const PostCard = ({ post }) => (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Avatar className="w-10 h-10">
              <AvatarImage src={post.user.avatar} />
              <AvatarFallback>{post.user.name.charAt(0)}</AvatarFallback>
            </Avatar>
            <div>
              <div className="flex items-center space-x-2">
                <p className="font-medium">{post.user.name}</p>
                {post.user.isFollowing && (
                  <Badge variant="secondary" className="text-xs">Following</Badge>
                )}
              </div>
              <p className="text-sm text-gray-600">
                with {post.pet.name} • {post.time}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm">
            <MapPin className="w-4 h-4 mr-1" />
            {post.location}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <p className="mb-4 text-gray-700">{post.content}</p>
        
        {/* Activity Info */}
        <div className="flex items-center space-x-4 mb-4 text-sm text-gray-600">
          <div className="flex items-center space-x-1">
            <Paw className="w-4 h-4" />
            <span>{post.activity.type}</span>
          </div>
          <div className="flex items-center space-x-1">
            <Clock className="w-4 h-4" />
            <span>{post.activity.duration} min</span>
          </div>
          <div className="flex items-center space-x-1">
            <MapPin className="w-4 h-4" />
            <span>{post.activity.distance} km</span>
          </div>
          <div className="flex items-center space-x-1">
            <Sun className="w-4 h-4" />
            <span>{post.weather.temp}°C</span>
          </div>
        </div>

        {/* Image Placeholder */}
        <div className="rounded-lg overflow-hidden mb-4">
          <div className="w-full h-64 bg-gradient-to-br from-blue-400 via-purple-500 to-pink-500 flex items-center justify-center">
            <Camera className="w-16 h-16 text-white" />
          </div>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-4">
          {post.tags.map((tag, index) => (
            <Badge key={index} variant="outline" className="text-xs">
              {tag}
            </Badge>
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between">
          <div className="flex space-x-4">
            <Button 
              variant="ghost" 
              size="sm" 
              className={`transition-colors ${likedPosts.has(post.id) ? 'text-red-500' : 'text-gray-600'}`}
              onClick={() => handleLike(post.id)}
            >
              <Heart className={`w-4 h-4 mr-1 ${likedPosts.has(post.id) ? 'fill-current' : ''}`} />
              {post.likes + (likedPosts.has(post.id) ? 1 : 0)}
            </Button>
            <Button variant="ghost" size="sm" className="text-gray-600">
              <MessageCircle className="w-4 h-4 mr-1" />
              {post.comments}
            </Button>
            <Button variant="ghost" size="sm" className="text-gray-600">
              <Share2 className="w-4 h-4 mr-1" />
              {post.shares}
            </Button>
          </div>
          <Badge variant="outline">
            {post.pet.breed}
          </Badge>
        </div>
      </CardContent>
    </Card>
  )

  const CreatePostForm = () => (
    <div className="space-y-4">
      <div>
        <Textarea 
          placeholder="Share your pet adventure..." 
          className="min-h-24 resize-none"
        />
      </div>
      
      <div className="flex items-center space-x-4">
        <Button variant="outline" size="sm" className="flex-1">
          <Camera className="w-4 h-4 mr-2" />
          Add Photo
        </Button>
        <Button variant="outline" size="sm" className="flex-1">
          <MapPin className="w-4 h-4 mr-2" />
          Add Location
        </Button>
        <Button variant="outline" size="sm" className="flex-1">
          <Paw className="w-4 h-4 mr-2" />
          Add Activity
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-sm font-medium">Pet</label>
          <select className="w-full mt-1 p-2 border rounded-md">
            <option>Buddy</option>
            <option>Whiskers</option>
          </select>
        </div>
        <div>
          <label className="text-sm font-medium">Mood</label>
          <select className="w-full mt-1 p-2 border rounded-md">
            <option>Happy</option>
            <option>Excited</option>
            <option>Proud</option>
            <option>Calm</option>
          </select>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Smile className="w-4 h-4 text-gray-600" />
          <span className="text-sm text-gray-600">Add to your story</span>
        </div>
        <Button className="bg-gradient-to-r from-blue-500 to-purple-600">
          Share Post
        </Button>
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
              <span className="text-xl font-bold text-gray-900">Social</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <Input
                  placeholder="Search posts..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 w-64"
                />
              </div>
              <Dialog open={isCreatePostOpen} onOpenChange={setIsCreatePostOpen}>
                <DialogTrigger asChild>
                  <Button>
                    <Plus className="w-4 h-4 mr-2" />
                    Create Post
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-lg">
                  <DialogHeader>
                    <DialogTitle>Create a Post</DialogTitle>
                    <DialogDescription>Share your pet adventure with the community</DialogDescription>
                  </DialogHeader>
                  <CreatePostForm />
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="feed">Feed</TabsTrigger>
            <TabsTrigger value="trending">Trending</TabsTrigger>
            <TabsTrigger value="friends">Friends</TabsTrigger>
          </TabsList>
          
          <TabsContent value="feed" className="space-y-6">
            {/* Create Post Card */}
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center space-x-3">
                  <Avatar className="w-12 h-12">
                    <AvatarImage src="/api/placeholder/48/48" />
                    <AvatarFallback>JD</AvatarFallback>
                  </Avatar>
                  <Dialog open={isCreatePostOpen} onOpenChange={setIsCreatePostOpen}>
                    <DialogTrigger asChild>
                      <Button variant="outline" className="flex-1 justify-start text-gray-600">
                        What adventures did you have today?
                      </Button>
                    </DialogTrigger>
                  </Dialog>
                </div>
              </CardContent>
            </Card>

            {/* Filter Options */}
            <div className="flex items-center justify-between">
              <div className="flex space-x-2">
                <Button variant="outline" size="sm">
                  <Filter className="w-4 h-4 mr-1" />
                  All Posts
                </Button>
                <Button variant="ghost" size="sm">Following</Button>
                <Button variant="ghost" size="sm">Nearby</Button>
              </div>
              <Select defaultValue="recent">
                <SelectTrigger className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="recent">Most Recent</SelectItem>
                  <SelectItem value="popular">Most Popular</SelectItem>
                  <SelectItem value="following">Following</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Social Feed */}
            <div className="space-y-4">
              {socialFeed.map((post) => (
                <PostCard key={post.id} post={post} />
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="trending" className="space-y-6">
            <h2 className="text-2xl font-bold">Trending Now</h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Trending Topics */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Trending Topics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {trendingTopics.map((topic, index) => (
                      <div key={index} className="flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg cursor-pointer">
                        <div>
                          <p className="font-medium">{topic.topic}</p>
                          <p className="text-sm text-gray-600">{topic.posts} posts</p>
                        </div>
                        <Badge variant="secondary">{topic.change}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Popular Posts */}
              <Card>
                <CardHeader>
                  <CardTitle>Popular This Week</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {socialFeed.slice(0, 3).map((post) => (
                      <div key={post.id} className="flex items-center space-x-3 p-3 hover:bg-gray-50 rounded-lg cursor-pointer">
                        <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-purple-500 rounded-lg flex items-center justify-center">
                          <Camera className="w-6 h-6 text-white" />
                        </div>
                        <div className="flex-1">
                          <p className="font-medium text-sm line-clamp-2">{post.content}</p>
                          <div className="flex items-center space-x-2 mt-1">
                            <Avatar className="w-6 h-6">
                              <AvatarImage src={post.user.avatar} />
                              <AvatarFallback className="text-xs">{post.user.name.charAt(0)}</AvatarFallback>
                            </Avatar>
                            <span className="text-xs text-gray-600">{post.user.name}</span>
                            <span className="text-xs text-gray-400">•</span>
                            <span className="text-xs text-gray-600">{post.likes} likes</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="friends" className="space-y-6">
            <h2 className="text-2xl font-bold">Friends Activity</h2>
            
            {/* Friends Activity Feed */}
            <div className="space-y-4">
              {friendsActivity.map((activity) => (
                <Card key={activity.id}>
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-3">
                      <Avatar className="w-12 h-12">
                        <AvatarImage src={activity.user.avatar} />
                        <AvatarFallback>{activity.user.name.charAt(0)}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <p className="font-medium">{activity.user.name}</p>
                        <p className="text-gray-600">
                          <span className="font-medium">{activity.pet.name}</span> {activity.activity.toLowerCase()}
                        </p>
                        <div className="flex items-center space-x-2 mt-1">
                          <MapPin className="w-3 h-3 text-gray-400" />
                          <span className="text-xs text-gray-500">{activity.location}</span>
                          <span className="text-xs text-gray-400">•</span>
                          <span className="text-xs text-gray-500">{activity.time}</span>
                        </div>
                      </div>
                      <Button variant="outline" size="sm">
                        <Heart className="w-4 h-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Friend Suggestions */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  Suggested Friends
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Array.from({ length: 4 }, (_, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <Avatar className="w-10 h-10">
                          <AvatarImage src={`/api/placeholder/40/40`} />
                          <AvatarFallback>U{i + 1}</AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="font-medium">User {i + 1}</p>
                          <p className="text-sm text-gray-600">Golden Retriever owner</p>
                        </div>
                      </div>
                      <Button size="sm">Follow</Button>
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