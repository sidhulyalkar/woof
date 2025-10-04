import React, { useState } from 'react';
import { Search, Plus, MoreHorizontal, MessageCircle, Users, Bell } from 'lucide-react';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

const mockConversations = [
  {
    id: 1,
    user: { name: 'Emma Davis', avatar: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Luna', type: 'Border Collie' },
    lastMessage: 'Luna did amazing at agility class today!',
    timestamp: '2m',
    unread: 2,
    online: true
  },
  {
    id: 2,
    user: { name: 'Mike Chen', avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Whiskers', type: 'Maine Coon' },
    lastMessage: 'Want to set up a playdate this weekend?',
    timestamp: '1h',
    unread: 0,
    online: true
  },
  {
    id: 3,
    user: { name: 'Sarah Johnson', avatar: 'https://images.unsplash.com/photo-1494790108755-2616b612b5c5?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Buddy', type: 'Golden Retriever' },
    lastMessage: 'Thanks for the walking tips!',
    timestamp: '3h',
    unread: 0,
    online: false
  },
  {
    id: 4,
    user: { name: 'Alex Rodriguez', avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Shadow', type: 'Belgian Malinois' },
    lastMessage: 'See you at the dog park tomorrow!',
    timestamp: '1d',
    unread: 1,
    online: false
  },
  {
    id: 5,
    user: { name: 'Lisa Chen', avatar: 'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Bella', type: 'Siberian Husky' },
    lastMessage: 'Bella loved meeting your pup!',
    timestamp: '2d',
    unread: 0,
    online: false
  }
];

const mockFriendRequests = [
  {
    id: 1,
    user: { name: 'David Miller', avatar: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Thunder', type: 'Australian Shepherd' },
    mutualFriends: 3
  },
  {
    id: 2,
    user: { name: 'Rachel Green', avatar: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&h=150&fit=crop&crop=face' },
    pet: { name: 'Coco', type: 'French Bulldog' },
    mutualFriends: 1
  }
];

export function MessagesScreen() {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('messages');

  const filteredConversations = mockConversations.filter(conversation =>
    conversation.user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    conversation.pet.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="bg-background min-h-screen">
      {/* Header */}
      <div className="sticky top-0 z-50 glass-card border-b border-border/20 px-4 py-4">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold">Messages</h1>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="p-2 rounded-full">
              <Bell size={20} />
            </Button>
            <Button size="sm" className="rounded-full bg-accent hover:bg-accent/90">
              <Plus size={20} />
            </Button>
          </div>
        </div>
        
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" size={16} />
          <Input
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 glass bg-surface-elevated/50 border-border/30"
          />
        </div>
      </div>

      <div className="pb-24">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="px-4 pt-4">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="messages" className="flex items-center gap-2">
              <MessageCircle size={16} />
              Messages
            </TabsTrigger>
            <TabsTrigger value="requests" className="flex items-center gap-2">
              <Users size={16} />
              Requests
              {mockFriendRequests.length > 0 && (
                <Badge variant="destructive" className="text-xs h-5 w-5 rounded-full p-0 flex items-center justify-center ml-1">
                  {mockFriendRequests.length}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="messages" className="mt-6">
            <div className="space-y-2">
              {filteredConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className="glass-card p-4 rounded-xl hover:bg-surface-elevated/80 transition-all duration-300 cursor-pointer group"
                >
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <Avatar className="w-14 h-14 ring-2 ring-accent/20">
                        <AvatarImage src={conversation.user.avatar} />
                        <AvatarFallback className="bg-accent/20">{conversation.user.name[0]}</AvatarFallback>
                      </Avatar>
                      {conversation.online && (
                        <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-success rounded-full border-2 border-background animate-pulse"></div>
                      )}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold truncate">{conversation.user.name}</span>
                        <Badge 
                          variant="secondary" 
                          className="text-xs shrink-0 bg-accent/20 text-accent border-accent/30"
                        >
                          {conversation.pet.name}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground truncate">
                        {conversation.lastMessage}
                      </p>
                    </div>
                    
                    <div className="flex flex-col items-end gap-2">
                      <span className="text-xs text-muted-foreground">
                        {conversation.timestamp}
                      </span>
                      <div className="flex items-center gap-2">
                        {conversation.unread > 0 && (
                          <Badge variant="destructive" className="text-xs h-6 w-6 rounded-full p-0 flex items-center justify-center">
                            {conversation.unread}
                          </Badge>
                        )}
                        <Button variant="ghost" size="sm" className="p-1 h-auto opacity-0 group-hover:opacity-100 transition-opacity">
                          <MoreHorizontal size={16} />
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {filteredConversations.length === 0 && searchQuery && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="w-16 h-16 bg-muted/30 rounded-full flex items-center justify-center mb-4">
                  <Search size={24} className="text-muted-foreground" />
                </div>
                <p className="text-lg font-medium mb-2">No conversations found</p>
                <p className="text-sm text-muted-foreground">Try searching for a different name or pet</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="requests" className="mt-6">
            {mockFriendRequests.length > 0 ? (
              <div className="space-y-4">
                {mockFriendRequests.map((request) => (
                  <div key={request.id} className="glass-card p-4 rounded-xl">
                    <div className="flex items-center gap-4">
                      <Avatar className="w-14 h-14 ring-2 ring-accent/20">
                        <AvatarImage src={request.user.avatar} />
                        <AvatarFallback className="bg-accent/20">{request.user.name[0]}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-semibold">{request.user.name}</span>
                          <Badge 
                            variant="secondary" 
                            className="text-xs bg-accent/20 text-accent border-accent/30"
                          >
                            {request.pet.name}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-1">
                          {request.pet.type}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {request.mutualFriends} mutual friends
                        </p>
                      </div>
                      <div className="flex flex-col gap-2">
                        <Button size="sm" className="bg-accent hover:bg-accent/90">
                          Accept
                        </Button>
                        <Button size="sm" variant="outline">
                          Ignore
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="w-16 h-16 bg-muted/30 rounded-full flex items-center justify-center mb-4">
                  <Users size={24} className="text-muted-foreground" />
                </div>
                <p className="text-lg font-medium mb-2">No friend requests</p>
                <p className="text-sm text-muted-foreground">When people want to connect, they'll appear here</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}