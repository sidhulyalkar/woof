import React, { useState } from 'react';
import { Heart, MessageCircle, Share2, MoreHorizontal, Plus, Search, Bell } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ImageWithFallback } from '@/components/figma/ImageWithFallback';

const mockPosts = [
  {
    id: 1,
    user: { name: 'Sarah Johnson', avatar: 'https://images.unsplash.com/photo-1494790108755-2616b612b5c5?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Buddy', type: 'Golden Retriever', badge: 'Active Buddy' },
    image: 'https://images.unsplash.com/photo-1758776217975-fd8415a2cb0c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxnb2xkZW4lMjByZXRyaWV2ZXIlMjBkb2clMjBwYXJrJTIwcGxheWluZ3xlbnwxfHx8fDE3NTkxOTkwNjJ8MA&ixlib=rb-4.1.0&q=80&w=1080',
    caption: 'Buddy had the best day at the dog park today! 🐕 We ran 2.5 miles together and made 3 new furry friends. Nothing beats seeing that happy golden smile! #happydog #dogpark #petfitness',
    likes: 142,
    comments: 23,
    timeAgo: '2h',
    activity: { distance: '2.5 mi', duration: '35 min' }
  },
  {
    id: 2,
    user: { name: 'Mike Chen', avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Whiskers', type: 'Maine Coon', badge: 'Zen Master' },
    image: 'https://images.unsplash.com/photo-1704624520371-5c4711339f57?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtYWluZSUyMGNvb24lMjBjYXQlMjBzdHJldGNoaW5nJTIwbW9ybmluZ3xlbnwxfHx8fDE3NTkxOTkwNjV8MA&ixlib=rb-4.1.0&q=80&w=1080',
    caption: 'Morning stretches with Whiskers 🐱 This cat knows how to start the day right! Look at that perfect form - teaching me a thing or two about flexibility.',
    likes: 89,
    comments: 12,
    timeAgo: '4h'
  },
  {
    id: 3,
    user: { name: 'Emma Davis', avatar: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Luna', type: 'Border Collie', badge: 'Agility Star' },
    image: 'https://images.unsplash.com/photo-1640958904594-fd2ed0b00167?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxib3JkZXIlMjBjb2xsaWUlMjBhZ2lsaXR5JTIwY291cnNlJTIwanVtcGluZ3xlbnwxfHx8fDE3NTkxOTkwNjh8MA&ixlib=rb-4.1.0&q=80&w=1080',
    caption: 'Luna completed her first agility course today! So proud of this smart girl ⭐ Clean run with perfect timing. Next stop: regional championships!',
    likes: 256,
    comments: 34,
    timeAgo: '6h',
    activity: { course: 'Level 2 Agility', score: '98/100' }
  }
];

export function FeedScreen() {
  const [likedPosts, setLikedPosts] = useState<Set<number>>(new Set());

  const handleLike = (postId: number) => {
    setLikedPosts(prev => {
      const newLiked = new Set(prev);
      if (newLiked.has(postId)) {
        newLiked.delete(postId);
      } else {
        newLiked.add(postId);
      }
      return newLiked;
    });
  };

  return (
    <div className="bg-background min-h-screen overflow-y-auto">
      {/* Header - Glass Morphism */}
      <div className="sticky top-0 z-50 glass-card border-b border-border/20 px-4 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-accent to-blue-400 bg-clip-text text-transparent">
            PetPath
          </h1>
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" className="p-2 rounded-full">
              <Search size={20} />
            </Button>
            <Button variant="ghost" size="sm" className="p-2 rounded-full relative">
              <Bell size={20} />
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-destructive rounded-full"></div>
            </Button>
            <Button size="sm" className="rounded-full bg-accent hover:bg-accent/90">
              <Plus size={20} />
            </Button>
          </div>
        </div>
      </div>

      {/* Feed */}
      <div className="pb-24 px-1">
        {mockPosts.map((post) => (
          <div key={post.id} className="glass-card mx-4 mb-6 rounded-xl overflow-hidden">
            {/* Post Header */}
            <div className="flex items-center justify-between px-4 py-4">
              <div className="flex items-center gap-3">
                <Avatar className="w-12 h-12 ring-2 ring-accent/20">
                  <AvatarImage src={post.user.avatar} />
                  <AvatarFallback className="bg-accent/20">{post.user.name[0]}</AvatarFallback>
                </Avatar>
                <div>
                  <div className="flex items-center gap-2">
                    <p className="font-semibold">{post.pet.name}</p>
                    {post.pet.badge && (
                      <Badge variant="secondary" className="text-xs bg-accent/20 text-accent border-accent/30">
                        {post.pet.badge}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    with {post.user.name} • {post.pet.type} • {post.timeAgo}
                  </p>
                </div>
              </div>
              <Button variant="ghost" size="sm" className="rounded-full">
                <MoreHorizontal size={20} />
              </Button>
            </div>

            {/* Post Image */}
            <div className="relative">
              <div className="aspect-square bg-muted/10">
                <ImageWithFallback
                  src={post.image}
                  alt={`${post.pet.name} the ${post.pet.type}`}
                  className="w-full h-full object-cover"
                />
              </div>
              {/* Activity overlay for fitness posts */}
              {post.activity && (
                <div className="absolute bottom-4 left-4 glass-card px-3 py-2 rounded-lg">
                  <div className="flex items-center gap-3 text-sm">
                    {post.activity.distance && (
                      <span className="text-accent font-medium">{post.activity.distance}</span>
                    )}
                    {post.activity.duration && (
                      <span className="text-muted-foreground">{post.activity.duration}</span>
                    )}
                    {post.activity.course && (
                      <span className="text-accent font-medium">{post.activity.course}</span>
                    )}
                    {post.activity.score && (
                      <span className="text-success font-medium">{post.activity.score}</span>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Post Actions */}
            <div className="px-4 py-4">
              <div className="flex items-center gap-6 mb-3">
                <Button
                  variant="ghost"
                  size="sm"
                  className="p-0 h-auto group"
                  onClick={() => handleLike(post.id)}
                >
                  <Heart
                    size={26}
                    className={`transition-all duration-300 ${
                      likedPosts.has(post.id)
                        ? 'fill-red-500 text-red-500 scale-110'
                        : 'text-foreground hover:text-red-400 group-hover:scale-110'
                    }`}
                  />
                </Button>
                <Button variant="ghost" size="sm" className="p-0 h-auto group">
                  <MessageCircle 
                    size={26} 
                    className="transition-all duration-300 group-hover:scale-110 hover:text-accent" 
                  />
                </Button>
                <Button variant="ghost" size="sm" className="p-0 h-auto group">
                  <Share2 
                    size={26} 
                    className="transition-all duration-300 group-hover:scale-110 hover:text-accent" 
                  />
                </Button>
              </div>

              <p className="text-sm mb-2 font-medium">
                {likedPosts.has(post.id) ? post.likes + 1 : post.likes} likes
              </p>

              <div className="text-sm leading-relaxed">
                <span className="font-semibold">{post.pet.name}</span>
                <span className="ml-2">{post.caption}</span>
              </div>

              {post.comments > 0 && (
                <button className="text-sm text-muted-foreground mt-2 hover:text-foreground transition-colors">
                  View all {post.comments} comments
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}