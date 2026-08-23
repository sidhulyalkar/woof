'use client';

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, RefreshCw, Search } from 'lucide-react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { ChatWindow } from '@/components/inbox/chat-window';
import { ConversationList } from '@/components/inbox/conversation-list';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { authApi } from '@/lib/api';
import { chatApi } from '@/lib/api/chat';
import { useAuthStore } from '@/lib/stores/auth-store';

export default function InboxPage() {
  const searchParams = useSearchParams();
  const memberParam = searchParams.get('member');
  const cachedUser = useAuthStore((state) => state.user);
  const queryClient = useQueryClient();
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [startError, setStartError] = useState<string | null>(null);
  const requestedMemberRef = useRef<string | null>(null);

  const profile = useQuery({
    queryKey: ['auth-profile'],
    queryFn: authApi.me,
    staleTime: 30_000,
    retry: false,
  });
  const user = profile.data ?? cachedUser;

  const conversations = useQuery({
    queryKey: ['chat', 'conversations'],
    queryFn: chatApi.getConversations,
    enabled: Boolean(user),
    staleTime: 15_000,
    retry: false,
  });

  useEffect(() => {
    if (!memberParam || !user || requestedMemberRef.current === memberParam) return;
    requestedMemberRef.current = memberParam;
    setStartError(null);

    void chatApi
      .createConversation(memberParam)
      .then(async ({ id }) => {
        await queryClient.invalidateQueries({ queryKey: ['chat', 'conversations'] });
        setSelectedConversation(id);
      })
      .catch(() => {
        requestedMemberRef.current = null;
        setStartError(
          'That conversation could not be opened. The member may no longer be available.'
        );
      });
  }, [memberParam, queryClient, user]);

  const filteredConversations = (conversations.data ?? []).filter((conversation) => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return true;
    return [conversation.participant.name, conversation.participant.petName ?? ''].some((value) =>
      value.toLowerCase().includes(query)
    );
  });

  const activeConversation = (conversations.data ?? []).find(
    (conversation) => conversation.id === selectedConversation
  );
  const unreadCount = (conversations.data ?? []).reduce(
    (total, conversation) => total + conversation.unreadCount,
    0
  );

  return (
    <div className="min-h-screen pb-20">
      <header className="glass-strong sticky top-0 z-40 border-b border-border/50">
        <div className="mx-auto max-w-lg space-y-3 px-4 py-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="eyebrow">Private coordination</p>
              <h1 className="mt-1 text-2xl font-bold">Messages</h1>
            </div>
            <div className="flex gap-2">
              <Button asChild size="sm" variant="outline" className="bg-transparent">
                <Link href="/meetups">Meetups</Link>
              </Button>
              <Button asChild size="sm">
                <Link href="/discover">Discover</Link>
              </Button>
            </div>
          </div>

          {!selectedConversation && (
            <div className="relative">
              <Search
                className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
                aria-hidden="true"
              />
              <Input
                placeholder={
                  unreadCount > 0 ? `Search · ${unreadCount} unread` : 'Search conversations'
                }
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                className="pl-9"
              />
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-lg">
        {startError && (
          <div
            className="mx-4 mt-4 rounded-2xl border border-destructive/20 bg-destructive/5 p-4 text-sm text-destructive"
            role="alert"
          >
            {startError}
          </div>
        )}

        {!user || conversations.isLoading ? (
          <div className="flex min-h-72 items-center justify-center gap-3" role="status">
            <Loader2 className="h-6 w-6 animate-spin text-primary" aria-hidden="true" />
            <span className="text-sm text-muted-foreground">Loading private conversations…</span>
          </div>
        ) : conversations.isError ? (
          <div className="surface-soft mx-4 mt-5 rounded-2xl p-6 text-center">
            <h2 className="font-semibold">Messages are temporarily unavailable</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Woof will not substitute demo conversations while canonical chat cannot be read.
            </p>
            <Button
              variant="outline"
              className="mt-4 gap-2 bg-transparent"
              onClick={() => conversations.refetch()}
            >
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
              Try again
            </Button>
          </div>
        ) : selectedConversation && activeConversation ? (
          <ChatWindow
            conversation={activeConversation}
            currentUserId={user.id}
            onBack={() => setSelectedConversation(null)}
          />
        ) : (
          <ConversationList
            conversations={filteredConversations}
            currentUserId={user.id}
            onSelectConversation={setSelectedConversation}
          />
        )}
      </main>

      <BottomNav />
    </div>
  );
}
