'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Send, Image as ImageIcon, Smile, MoreVertical } from 'lucide-react';
import { ProfileAvatar, getPlaceholderAvatar } from '@/components/ui/ProfileAvatar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

// Mock data - replace with real data from API
const mockConversation = {
  id: 1,
  user: { name: 'Emma Davis', id: 'user-emma' },
  pet: { name: 'Luna', type: 'Border Collie' },
  messages: [
    {
      id: 1,
      senderId: 'user-emma',
      text: 'Hey! Luna did amazing at agility class today!',
      timestamp: '10:32 AM',
      isOwn: false,
    },
    {
      id: 2,
      senderId: 'current-user',
      text: 'That's awesome! How did she do with the weave poles?',
      timestamp: '10:34 AM',
      isOwn: true,
    },
    {
      id: 3,
      senderId: 'user-emma',
      text: 'She nailed them! Her trainer was so impressed',
      timestamp: '10:35 AM',
      isOwn: false,
    },
    {
      id: 4,
      senderId: 'current-user',
      text: 'Amazing! We should set up a playdate this weekend',
      timestamp: '10:37 AM',
      isOwn: true,
    },
    {
      id: 5,
      senderId: 'user-emma',
      text: 'Yes! Saturday afternoon works great for us',
      timestamp: '10:38 AM',
      isOwn: false,
    },
  ],
};

export default function ConversationPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const [message, setMessage] = useState('');

  const handleSendMessage = () => {
    if (!message.trim()) return;
    // TODO: Send message via API
    console.log('Sending message:', message);
    setMessage('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50/30 flex flex-col">
      {/* Header */}
      <div className="sticky top-0 z-50 backdrop-blur-2xl bg-white/90 border-b border-gray-200/40 px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.back()}
              className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-gray-100 active:scale-95 transition-all duration-200"
            >
              <ArrowLeft className="h-5 w-5 text-gray-700" />
            </button>

            <div className="flex items-center gap-3">
              <ProfileAvatar
                type="user"
                src={getPlaceholderAvatar(mockConversation.user.name, 'user')}
                fallbackText={mockConversation.user.name.slice(0, 2).toUpperCase()}
                size="md"
              />
              <div>
                <h2 className="font-bold text-[15px] text-gray-900">{mockConversation.user.name}</h2>
                <p className="text-sm text-gray-600">{mockConversation.pet.name}</p>
              </div>
            </div>
          </div>

          <button className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-gray-100 transition-colors">
            <MoreVertical className="h-5 w-5 text-gray-700" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {mockConversation.messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.isOwn ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex gap-2 max-w-[75%] ${msg.isOwn ? 'flex-row-reverse' : 'flex-row'}`}>
                {!msg.isOwn && (
                  <ProfileAvatar
                    type="user"
                    src={getPlaceholderAvatar(mockConversation.user.name, 'user')}
                    fallbackText={mockConversation.user.name.slice(0, 2).toUpperCase()}
                    size="sm"
                    className="shrink-0"
                  />
                )}
                <div>
                  <div
                    className={`px-4 py-3 rounded-3xl ${
                      msg.isOwn
                        ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white'
                        : 'bg-white/80 backdrop-blur-xl border border-gray-200/60 text-gray-900'
                    }`}
                  >
                    <p className="text-[15px] leading-5">{msg.text}</p>
                  </div>
                  <p className={`text-xs text-gray-500 mt-1 px-2 ${msg.isOwn ? 'text-right' : 'text-left'}`}>
                    {msg.timestamp}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="sticky bottom-0 backdrop-blur-2xl bg-white/90 border-t border-gray-200/40 px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center gap-2">
          <button className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-gray-100 transition-colors shrink-0">
            <ImageIcon className="h-5 w-5 text-gray-600" />
          </button>

          <div className="flex-1 relative">
            <Input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder="Message..."
              className="pr-12 bg-gray-50 border-gray-200 rounded-3xl h-11 focus-visible:ring-blue-500/30"
            />
            <button className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-200 transition-colors">
              <Smile className="h-5 w-5 text-gray-600" />
            </button>
          </div>

          <Button
            onClick={handleSendMessage}
            disabled={!message.trim()}
            className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 p-0 disabled:opacity-30 shrink-0"
          >
            <Send className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
