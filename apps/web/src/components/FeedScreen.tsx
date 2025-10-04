'use client';

import { useRouter } from 'next/navigation';
import { Plus, Search, Loader2, Bell } from 'lucide-react';
import { PostCard } from '@/components/feed/PostCard';
import { StoryCircle } from '@/components/feed/StoryCircle';
import { useFeed } from '@/lib/api/hooks';
import { useSessionStore } from '@/store/session';
import { useUIStore } from '@/store/ui';

export function FeedScreen() {
  const router = useRouter();
  const { data: posts, isLoading, error } = useFeed();
  const { user } = useSessionStore();
  const { showToast } = useUIStore();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50/30">
      {/* Premium Instagram-Style Header */}
      <header className="sticky top-0 z-50 backdrop-blur-2xl bg-white/90 border-b border-gray-200/40">
        <div className="max-w-3xl mx-auto flex items-center justify-between px-6 h-16">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Woof
          </h1>
          <div className="flex items-center gap-3">
            <button className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-gray-100 active:scale-95 transition-all duration-200">
              <Search className="h-5 w-5 text-gray-700" />
            </button>
            <button className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-gray-100 active:scale-95 transition-all duration-200 relative">
              <Bell className="h-5 w-5 text-gray-700" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full ring-2 ring-white"></span>
            </button>
            <button
              onClick={() => router.push('/create-post')}
              className="w-10 h-10 flex items-center justify-center rounded-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 shadow-lg hover:shadow-xl active:scale-95 transition-all duration-200"
            >
              <Plus className="h-5 w-5 text-white" strokeWidth={2.5} />
            </button>
          </div>
        </div>
      </header>

      {/* Instagram-Style Stories */}
      <div className="border-b border-gray-200/30 bg-white/70 backdrop-blur-md">
        <div className="max-w-3xl mx-auto flex gap-4 overflow-x-auto scrollbar-hide px-6 py-4">
          <StoryCircle
            username={user?.username || 'You'}
            avatar={user?.avatar}
            isOwn
            onClick={() => showToast({ message: 'Story creation coming soon!', type: 'info' })}
          />
          <StoryCircle
            username="buddy123"
            avatar="https://images.unsplash.com/photo-1600481176431-47ad15e648a7?w=150&h=150&fit=crop"
            hasStory
            onClick={() => showToast({ message: 'Story viewer coming soon!', type: 'info' })}
          />
          <StoryCircle
            username="luna_pup"
            avatar="https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=150&h=150&fit=crop"
            hasStory
            onClick={() => showToast({ message: 'Story viewer coming soon!', type: 'info' })}
          />
        </div>
      </div>

      {/* Premium Instagram-Style Feed */}
      <main className="pb-24 pt-6">
        <div className="max-w-3xl mx-auto px-6">
          {isLoading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            </div>
          )}

          {error && (
            <div className="p-12 text-center bg-white/90 backdrop-blur-xl border border-gray-200/40 rounded-3xl shadow-lg">
              <div className="w-16 h-16 rounded-full bg-red-50 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-base font-semibold text-gray-900 mb-2">Failed to load feed</p>
              <p className="text-sm text-gray-600 mb-6">Something went wrong. Please try again.</p>
              <button
                onClick={() => window.location.reload()}
                className="px-6 py-3 text-sm font-semibold text-white bg-gradient-to-r from-blue-500 to-blue-600 rounded-full hover:from-blue-600 hover:to-blue-700 shadow-lg hover:shadow-xl active:scale-95 transition-all duration-200"
              >
                Retry
              </button>
            </div>
          )}

          {!isLoading && !error && posts && posts.length === 0 && (
            <div className="p-16 text-center bg-white/90 backdrop-blur-xl border border-gray-200/40 rounded-3xl shadow-lg">
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-50 to-indigo-50 flex items-center justify-center mx-auto mb-6">
                <Plus className="h-10 w-10 text-blue-500" strokeWidth={2} />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">No posts yet</h3>
              <p className="text-base text-gray-600 mb-8 max-w-sm mx-auto">
                Be the first to share something amazing with the Woof community!
              </p>
              <button
                onClick={() => router.push('/create-post')}
                className="px-8 py-3.5 text-base font-semibold text-white bg-gradient-to-r from-blue-500 to-blue-600 rounded-full hover:from-blue-600 hover:to-blue-700 shadow-lg hover:shadow-xl active:scale-95 transition-all duration-200"
              >
                Create Your First Post
              </button>
            </div>
          )}

          {posts && posts.length > 0 && (
            <div className="space-y-6">
              {posts.map((post) => (
                <div key={post.id} className="bg-white/90 backdrop-blur-xl border border-gray-200/40 rounded-3xl shadow-lg hover:shadow-xl overflow-hidden transition-all duration-300">
                  <PostCard post={post} />
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
