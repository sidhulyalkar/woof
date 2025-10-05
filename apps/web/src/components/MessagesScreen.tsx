import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Search, Plus, MoreHorizontal, MessageCircle, Users, Bell } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { EmojiAvatar, getEmojiForId, getVariantForId } from '@/components/ui/EmojiAvatar';

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
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('messages');

  const filteredConversations = mockConversations.filter(conversation =>
    conversation.user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    conversation.pet.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleConversationClick = (conversationId: number) => {
    router.push(`/messages/${conversationId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50/30">
      {/* Header */}
      <div className="sticky top-0 z-50 backdrop-blur-2xl bg-white/90 border-b border-gray-200/40 px-6 py-4">
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
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
          <Input
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-gray-50 border-gray-200 rounded-xl h-11 focus-visible:ring-blue-500/30"
          />
        </div>
      </div>

      <div className="pb-24 max-w-3xl mx-auto">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="px-6 pt-4">
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
            <div className="space-y-3">
              {filteredConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  onClick={() => handleConversationClick(conversation.id)}
                  className="bg-white/80 backdrop-blur-xl border border-gray-200/60 p-4 rounded-3xl hover:shadow-md hover:bg-white transition-all duration-200 cursor-pointer active:scale-[0.98]"
                >
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <EmojiAvatar
                        emoji={getEmojiForId(conversation.user.name, 'user')}
                        variant={getVariantForId(conversation.user.name)}
                        size="lg"
                      />
                      {conversation.online && (
                        <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white shadow-sm"></div>
                      )}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-bold text-[15px] text-gray-900 truncate">{conversation.user.name}</span>
                        <span className="text-gray-300">·</span>
                        <span className="text-sm font-medium text-gray-600 truncate">{conversation.pet.name}</span>
                      </div>
                      <p className="text-[15px] text-gray-600 truncate leading-5">
                        {conversation.lastMessage}
                      </p>
                    </div>

                    <div className="flex flex-col items-end gap-2 shrink-0">
                      <span className="text-xs text-gray-500 font-medium">
                        {conversation.timestamp}
                      </span>
                      {conversation.unread > 0 && (
                        <div className="w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center">
                          <span className="text-white text-xs font-bold">{conversation.unread}</span>
                        </div>
                      )}
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
              <div className="space-y-3">
                {mockFriendRequests.map((request) => (
                  <div key={request.id} className="bg-white/80 backdrop-blur-xl border border-gray-200/60 p-4 rounded-3xl">
                    <div className="flex items-center gap-4">
                      <EmojiAvatar
                        emoji={getEmojiForId(request.user.name, 'user')}
                        variant={getVariantForId(request.user.name)}
                        size="lg"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-bold text-[15px] text-gray-900">{request.user.name}</span>
                          <span className="text-gray-300">·</span>
                          <span className="text-sm font-medium text-gray-600">{request.pet.name}</span>
                        </div>
                        <p className="text-sm text-gray-600 mb-1">
                          {request.pet.type}
                        </p>
                        <p className="text-xs text-gray-500">
                          {request.mutualFriends} mutual friends
                        </p>
                      </div>
                      <div className="flex flex-col gap-2 shrink-0">
                        <Button size="sm" className="bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 rounded-xl px-6">
                          Accept
                        </Button>
                        <Button size="sm" variant="outline" className="rounded-xl px-6 border-gray-200">
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