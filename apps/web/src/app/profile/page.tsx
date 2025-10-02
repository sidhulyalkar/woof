"use client";
import React, { useState } from 'react';
import { Settings, Edit3, Share2, MapPin, Calendar, Award, Users, Camera, Heart, MessageCircle, Star, Plus } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ImageWithFallback } from '@/components/figma/ImageWithFallback';

const userProfile = {
  name: 'Sarah Johnson',
  avatar: 'https://images.unsplash.com/photo-1494790108755-2616b612b5c5?w=150&h=150&fit=crop&crop=face',
  location: 'San Francisco, CA',
  joinDate: 'March 2023',
  bio: 'Dog lover and outdoor enthusiast. Always looking for new trails to explore with my furry friends!',
  stats: {
    posts: 127,
    friends: 245,
    totalDistance: '342 km',
    achievements: 12
  }
};

const userPets = [
  {
    id: 1,
    name: 'Buddy',
    type: 'Golden Retriever',
    age: '3 years',
    avatar: 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=150&h=150&fit=crop&crop=face',
    bio: 'Loves fetch and swimming. Champion belly rub receiver.',
    achievements: ['Best Walker', 'Social Butterfly', 'Park Explorer']
  },
  {
    id: 2,
    name: 'Luna',
    type: 'Border Collie Mix',
    age: '2 years',
    avatar: 'https://images.unsplash.com/photo-1551717743-49959800b1f6?w=150&h=150&fit=crop&crop=face',
    bio: 'Agility champion in training. Smart and energetic.',
    achievements: ['Speed Demon', 'Agility Star']
  }
];

const recentPosts = [
  {
    id: 1,
    image: 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=300&h=300&fit=crop',
    likes: 142,
    comments: 23
  },
  {
    id: 2,
    image: 'https://images.unsplash.com/photo-1551717743-49959800b1f6?w=300&h=300&fit=crop',
    likes: 89,
    comments: 12
  },
  {
    id: 3,
    image: 'https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=300&h=300&fit=crop',
    likes: 256,
    comments: 34
  },
  {
    id: 4,
    image: 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300&h=300&fit=crop',
    likes: 178,
    comments: 19
  },
  {
    id: 5,
    image: 'https://images.unsplash.com/photo-1605568427561-40dd23c2acea?w=300&h=300&fit=crop',
    likes: 203,
    comments: 28
  },
  {
    id: 6,
    image: 'https://images.unsplash.com/photo-1589941013453-ec89f33b5e95?w=300&h=300&fit=crop',
    likes: 156,
    comments: 15
  }
];

export default function ProfilePage() {
  const [activeTab, setActiveTab] = useState('posts');

  return (
    <div className="bg-background min-h-screen">
      {/* Header */}
      <div className="sticky top-0 z-50 glass-card border-b border-border/20 px-4 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Profile</h1>
          <Button variant="ghost" size="sm" className="p-2 rounded-full">
            <Settings size={20} />
          </Button>
        </div>
      </div>

      <div className="pb-24">
        {/* Profile Header */}
        <div className="relative px-4 py-8">
          {/* Background Pattern */}
          <div className="absolute inset-0 bg-gradient-to-br from-accent/10 to-transparent rounded-xl mx-4"></div>
          
          <div className="relative z-10 text-center">
            <div className="relative inline-block mb-6">
              <Avatar className="w-32 h-32 mx-auto ring-4 ring-accent/20">
                <AvatarImage src={userProfile.avatar} />
                <AvatarFallback className="bg-accent/20 text-2xl">{userProfile.name[0]}</AvatarFallback>
              </Avatar>
              <Button size="sm" className="absolute -bottom-2 -right-2 rounded-full w-10 h-10 p-0 bg-accent hover:bg-accent/90">
                <Camera size={16} />
              </Button>
            </div>
            
            <h2 className="text-2xl font-bold mb-2">{userProfile.name}</h2>
            
            <div className="flex items-center justify-center gap-6 text-sm text-muted-foreground mb-4">
              <div className="flex items-center gap-1">
                <MapPin size={16} />
                {userProfile.location}
              </div>
              <div className="flex items-center gap-1">
                <Calendar size={16} />
                Joined {userProfile.joinDate}
              </div>
            </div>
            
            <p className="text-muted-foreground mb-6 max-w-md mx-auto leading-relaxed">
              {userProfile.bio}
            </p>
            
            <div className="flex justify-center gap-3 mb-8">
              <Button size="sm" className="bg-accent hover:bg-accent/90">
                <Edit3 size={16} className="mr-2" />
                Edit Profile
              </Button>
              <Button size="sm" variant="outline">
                <Share2 size={16} className="mr-2" />
                Share
              </Button>
            </div>
            
            {/* Stats Cards */}
            <div className="grid grid-cols-4 gap-3">
              <div className="glass-card p-4 rounded-xl text-center">
                <p className="text-xl font-bold text-accent">{userProfile.stats.posts}</p>
                <p className="text-xs text-muted-foreground">Posts</p>
              </div>
              <div className="glass-card p-4 rounded-xl text-center">
                <p className="text-xl font-bold text-accent">{userProfile.stats.friends}</p>
                <p className="text-xs text-muted-foreground">Friends</p>
              </div>
              <div className="glass-card p-4 rounded-xl text-center">
                <p className="text-xl font-bold text-accent">{userProfile.stats.totalDistance}</p>
                <p className="text-xs text-muted-foreground">Distance</p>
              </div>
              <div className="glass-card p-4 rounded-xl text-center">
                <p className="text-xl font-bold text-accent">{userProfile.stats.achievements}</p>
                <p className="text-xs text-muted-foreground">Awards</p>
              </div>
            </div>
          </div>
        </div>

        {/* Content Tabs */}
        <div className="px-4">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3 mb-6">
              <TabsTrigger value="posts">Posts</TabsTrigger>
              <TabsTrigger value="pets">My Pets</TabsTrigger>
              <TabsTrigger value="activity">Activity</TabsTrigger>
            </TabsList>
            
            <TabsContent value="posts" className="space-y-4">
              <div className="grid grid-cols-3 gap-2">
                {recentPosts.map((post) => (
                  <div key={post.id} className="relative aspect-square glass-card rounded-xl overflow-hidden group">
                    <ImageWithFallback
                      src={post.image}
                      alt="Pet post"
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <div className="absolute bottom-2 left-2 flex items-center gap-3 text-white text-sm">
                        <div className="flex items-center gap-1">
                          <Heart size={14} />
                          {post.likes}
                        </div>
                        <div className="flex items-center gap-1">
                          <MessageCircle size={14} />
                          {post.comments}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </TabsContent>
            
            <TabsContent value="pets" className="space-y-4">
              {userPets.map((pet) => (
                <div key={pet.id} className="glass-card p-6 rounded-xl">
                  <div className="flex items-start gap-4">
                    <Avatar className="w-20 h-20 ring-2 ring-accent/20">
                      <AvatarImage src={pet.avatar} />
                      <AvatarFallback className="bg-accent/20 text-lg">{pet.name[0]}</AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h3 className="text-xl font-bold">{pet.name}</h3>
                        <Badge 
                          variant="secondary" 
                          className="bg-accent/20 text-accent border-accent/30"
                        >
                          {pet.type}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-3">{pet.age}</p>
                      <p className="text-sm mb-4 leading-relaxed">{pet.bio}</p>
                      <div className="flex flex-wrap gap-2">
                        {pet.achievements.map((achievement, index) => (
                          <Badge 
                            key={index} 
                            variant="secondary" 
                            className="text-xs bg-success/20 text-success border-success/30"
                          >
                            <Star size={10} className="mr-1" />
                            {achievement}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              
              <Button variant="outline" className="w-full glass-card border-dashed border-accent/30 hover:bg-accent/10">
                <Plus size={16} className="mr-2" />
                Add New Pet
              </Button>
            </TabsContent>
            
            <TabsContent value="activity" className="space-y-6">
              <div className="glass-card p-6 rounded-xl">
                <h3 className="text-lg font-semibold mb-4">This Month</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-surface-elevated/50 rounded-lg">
                    <p className="text-2xl font-bold text-accent">28.4 km</p>
                    <p className="text-sm text-muted-foreground">Distance</p>
                  </div>
                  <div className="text-center p-4 bg-surface-elevated/50 rounded-lg">
                    <p className="text-2xl font-bold text-success">1,840</p>
                    <p className="text-sm text-muted-foreground">Calories</p>
                  </div>
                  <div className="text-center p-4 bg-surface-elevated/50 rounded-lg">
                    <p className="text-2xl font-bold text-warning">15h 32m</p>
                    <p className="text-sm text-muted-foreground">Active Time</p>
                  </div>
                  <div className="text-center p-4 bg-surface-elevated/50 rounded-lg">
                    <p className="text-2xl font-bold text-accent">23</p>
                    <p className="text-sm text-muted-foreground">Activities</p>
                  </div>
                </div>
              </div>
              
              <div className="glass-card p-6 rounded-xl">
                <h3 className="text-lg font-semibold mb-4">Recent Achievements</h3>
                <div className="space-y-4">
                  <div className="flex items-center gap-4 p-3 bg-surface-elevated/30 rounded-lg">
                    <div className="w-12 h-12 bg-yellow-500/20 rounded-full flex items-center justify-center">
                      <Award className="text-yellow-500" size={20} />
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold">Marathon Master</p>
                      <p className="text-sm text-muted-foreground">Completed 100km this month</p>
                    </div>
                    <Badge variant="secondary" className="bg-yellow-500/20 text-yellow-500 border-yellow-500/30">
                      New!
                    </Badge>
                  </div>
                  <div className="flex items-center gap-4 p-3 bg-surface-elevated/30 rounded-lg">
                    <div className="w-12 h-12 bg-accent/20 rounded-full flex items-center justify-center">
                      <Users className="text-accent" size={20} />
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold">Social Butterfly</p>
                      <p className="text-sm text-muted-foreground">Made 10 new friends</p>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}