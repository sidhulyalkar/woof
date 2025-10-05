'use client';

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Avatar } from '@/components/ui/avatar';
import { MessageCircle, Calendar, Send, Sparkles, MapPin } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { toast } from 'sonner';
import { MeetupProposalScreen } from '../meetups/MeetupProposalScreen';

interface Match {
  id: string;
  userId: string;
  handle: string;
  avatarUrl?: string;
  lastMessage?: string;
  lastMessageTime?: string;
  unreadCount?: number;
  compatibilityScore?: number;
  distance?: number;
}

interface Message {
  id: string;
  senderId: string;
  text: string;
  createdAt: string;
  sender: {
    id: string;
    handle: string;
    avatarUrl?: string;
  };
}

export function EnhancedChatScreen() {
  const [selectedMatch, setSelectedMatch] = useState<Match | null>(null);
  const [showMeetupProposal, setShowMeetupProposal] = useState(false);

  const { data: matches, isLoading } = useQuery<Match[]>({
    queryKey: ['matches', 'conversations'],
    queryFn: async () => {
      // Get matches with conversations
      const data = await apiClient.get<Match[]>('/matches/with-conversations');
      return data;
    },
  });

  if (showMeetupProposal && selectedMatch) {
    return (
      <MeetupProposalScreen
        matchUserId={selectedMatch.userId}
        onClose={() => setShowMeetupProposal(false)}
      />
    );
  }

  if (selectedMatch) {
    return (
      <ChatConversation
        match={selectedMatch}
        onBack={() => setSelectedMatch(null)}
        onProposeMeetup={() => setShowMeetupProposal(true)}
      />
    );
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border/20">
        <div className="max-w-2xl mx-auto">
          <h1 className="text-2xl font-bold">Messages</h1>
          <p className="text-sm text-muted-foreground">Connect with your matches</p>
        </div>
      </div>

      {/* Matches List */}
      <div className="flex-1 overflow-auto">
        <div className="max-w-2xl mx-auto p-4 space-y-2">
          {isLoading ? (
            <div className="text-center py-8">
              <MessageCircle className="w-12 h-12 mx-auto mb-4 text-accent animate-pulse" />
              <p className="text-muted-foreground">Loading conversations...</p>
            </div>
          ) : !matches || matches.length === 0 ? (
            <Card className="p-8 text-center">
              <MessageCircle className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">No conversations yet</h3>
              <p className="text-muted-foreground">
                Start swiping to find matches and begin conversations!
              </p>
            </Card>
          ) : (
            matches.map(match => (
              <MatchCard
                key={match.id}
                match={match}
                onClick={() => setSelectedMatch(match)}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

interface MatchCardProps {
  match: Match;
  onClick: () => void;
}

function MatchCard({ match, onClick }: MatchCardProps) {
  return (
    <Card
      className="p-4 hover:shadow-md transition-all cursor-pointer"
      onClick={onClick}
    >
      <div className="flex items-center gap-3">
        <div className="relative">
          <Avatar className="w-12 h-12">
            {match.avatarUrl ? (
              <img src={match.avatarUrl} alt={match.handle} />
            ) : (
              <div className="bg-accent text-white flex items-center justify-center">
                {match.handle[0].toUpperCase()}
              </div>
            )}
          </Avatar>
          {match.unreadCount && match.unreadCount > 0 && (
            <div className="absolute -top-1 -right-1 w-5 h-5 bg-accent text-white text-xs rounded-full flex items-center justify-center">
              {match.unreadCount}
            </div>
          )}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1">
            <p className="font-semibold">@{match.handle}</p>
            {match.compatibilityScore && (
              <Badge variant="secondary" className="text-xs">
                {match.compatibilityScore}% match
              </Badge>
            )}
          </div>
          {match.lastMessage && (
            <p className="text-sm text-muted-foreground truncate">
              {match.lastMessage}
            </p>
          )}
          <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
            {match.distance && (
              <>
                <MapPin className="w-3 h-3" />
                <span>{match.distance < 1 ? `${(match.distance * 1000).toFixed(0)}m` : `${match.distance.toFixed(1)}km`}</span>
              </>
            )}
            {match.lastMessageTime && (
              <span>• {new Date(match.lastMessageTime).toLocaleDateString()}</span>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

interface ChatConversationProps {
  match: Match;
  onBack: () => void;
  onProposeMeetup: () => void;
}

function ChatConversation({ match, onBack, onProposeMeetup }: ChatConversationProps) {
  const [messageText, setMessageText] = useState('');
  const [showMeetupCTA, setShowMeetupCTA] = useState(true);
  const queryClient = useQueryClient();

  const { data: messages } = useQuery<Message[]>({
    queryKey: ['conversation', match.userId],
    queryFn: () => apiClient.get<Message[]>(`/chat/conversations/${match.userId}/messages`),
    refetchInterval: 3000, // Poll every 3 seconds
  });

  const sendMessageMutation = useMutation({
    mutationFn: (text: string) =>
      apiClient.post(`/chat/conversations/${match.userId}/messages`, { text }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversation', match.userId] });
      setMessageText('');
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to send message');
    },
  });

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (messageText.trim()) {
      sendMessageMutation.mutate(messageText);
    }
  };

  // Show meetup CTA after 5+ messages exchanged
  const shouldShowMeetupCTA = (messages?.length || 0) >= 5;

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border/20">
        <div className="max-w-2xl mx-auto flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={onBack}>
            ← Back
          </Button>
          <Avatar className="w-10 h-10">
            {match.avatarUrl ? (
              <img src={match.avatarUrl} alt={match.handle} />
            ) : (
              <div className="bg-accent text-white flex items-center justify-center">
                {match.handle[0].toUpperCase()}
              </div>
            )}
          </Avatar>
          <div className="flex-1">
            <p className="font-semibold">@{match.handle}</p>
            {match.compatibilityScore && (
              <p className="text-xs text-muted-foreground">
                {match.compatibilityScore}% compatible
              </p>
            )}
          </div>
          <Button size="sm" variant="outline" onClick={onProposeMeetup}>
            <Calendar className="w-4 h-4 mr-2" />
            Meetup
          </Button>
        </div>
      </div>

      {/* Meetup CTA Banner */}
      {shouldShowMeetupCTA && showMeetupCTA && (
        <div className="bg-gradient-to-r from-accent/20 to-primary/20 border-b border-accent/20">
          <div className="max-w-2xl mx-auto p-4">
            <div className="flex items-start gap-3">
              <Sparkles className="w-5 h-5 text-accent flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-semibold mb-1">Ready to meet IRL?</p>
                <p className="text-sm text-muted-foreground mb-3">
                  You've been chatting for a while! Time to plan a meetup?
                </p>
                <div className="flex gap-2">
                  <Button size="sm" onClick={onProposeMeetup}>
                    <Calendar className="w-4 h-4 mr-2" />
                    Propose Meetup
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setShowMeetupCTA(false)}
                  >
                    Maybe later
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-2xl mx-auto space-y-4">
          {!messages || messages.length === 0 ? (
            <div className="text-center py-8">
              <MessageCircle className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-muted-foreground">No messages yet. Say hi! 👋</p>
            </div>
          ) : (
            messages.map(message => (
              <MessageBubble key={message.id} message={message} />
            ))
          )}
        </div>
      </div>

      {/* Input */}
      <div className="p-4 border-t border-border/20">
        <div className="max-w-2xl mx-auto">
          <form onSubmit={handleSend} className="flex gap-2">
            <Input
              value={messageText}
              onChange={(e) => setMessageText(e.target.value)}
              placeholder="Type a message..."
              className="flex-1"
            />
            <Button
              type="submit"
              disabled={!messageText.trim() || sendMessageMutation.isPending}
            >
              <Send className="w-4 h-4" />
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}

interface MessageBubbleProps {
  message: Message;
}

function MessageBubble({ message }: MessageBubbleProps) {
  const isOwnMessage = false; // TODO: Check against current user ID

  return (
    <div className={`flex gap-3 ${isOwnMessage ? 'flex-row-reverse' : ''}`}>
      <Avatar className="w-8 h-8 flex-shrink-0">
        {message.sender.avatarUrl ? (
          <img src={message.sender.avatarUrl} alt={message.sender.handle} />
        ) : (
          <div className="bg-accent text-white flex items-center justify-center text-xs">
            {message.sender.handle[0].toUpperCase()}
          </div>
        )}
      </Avatar>
      <div className={`flex-1 max-w-[70%] ${isOwnMessage ? 'items-end' : ''}`}>
        <div
          className={`p-3 rounded-2xl ${
            isOwnMessage
              ? 'bg-accent text-white rounded-br-sm'
              : 'bg-muted rounded-bl-sm'
          }`}
        >
          <p className="text-sm">{message.text}</p>
        </div>
        <p className="text-xs text-muted-foreground mt-1 px-2">
          {new Date(message.createdAt).toLocaleTimeString([], {
            hour: 'numeric',
            minute: '2-digit',
          })}
        </p>
      </div>
    </div>
  );
}
