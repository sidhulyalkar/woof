'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Heart, MessageCircle, Share2, Send, MoreHorizontal, X } from 'lucide-react';
import { ProfileAvatar, getPlaceholderAvatar } from '@/components/ui/ProfileAvatar';
import { Input } from '@/components/ui/input';
import { formatDistanceToNow } from 'date-fns';
import { useUIStore } from '@/store/ui';

// Mock data - will be replaced with real API call
const mockPost = {
  id: '1',
  user: {
    id: 'user1',
    username: 'Shasta21',
    avatar: null,
  },
  pet: {
    name: 'Buddy',
    avatar: null,
  },
  content: 'hello',
  images: ['https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800'],
  likes: 0,
  comments: 0,
  createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
};

const mockComments = [
  {
    id: '1',
    authorUserId: 'user2',
    text: 'So cute!',
    createdAt: new Date().toISOString(),
    author: {
      id: 'user2',
      handle: 'JaneDoe',
      avatarUrl: null,
    },
  },
];

export default function PostDetailPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const { showToast } = useUIStore();
  const [comment, setComment] = useState('');
  const [isLiked, setIsLiked] = useState(false);
  const [showComments, setShowComments] = useState(false);

  const handleSendComment = () => {
    if (!comment.trim()) return;
    showToast({ message: 'Comment posted!', type: 'success' });
    setComment('');
  };

  return (
    <div className="min-h-screen bg-black">
      {/* Header */}
      <div className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-b from-black/80 to-transparent backdrop-blur-sm">
        <div className="flex items-center justify-between px-4 py-3">
          <button
            onClick={() => router.back()}
            className="w-10 h-10 flex items-center justify-center rounded-full bg-black/40 hover:bg-black/60 active:scale-95 transition-all"
          >
            <ArrowLeft className="h-5 w-5 text-white" />
          </button>
          <div className="flex items-center gap-3">
            <ProfileAvatar
              src={mockPost.user.avatar || getPlaceholderAvatar(mockPost.user.username, 'user')}
              alt={mockPost.user.username}
              type="user"
              size="sm"
              fallbackText={mockPost.user.username.slice(0, 2).toUpperCase()}
            />
            <span className="text-white font-semibold text-sm">{mockPost.user.username}</span>
          </div>
          <button className="w-10 h-10 flex items-center justify-center rounded-full bg-black/40 hover:bg-black/60 active:scale-95 transition-all">
            <MoreHorizontal className="h-5 w-5 text-white" />
          </button>
        </div>
      </div>

      {/* Image Container */}
      <div className="relative w-full h-screen flex items-center justify-center bg-black">
        {mockPost.images && mockPost.images.length > 0 && (
          <img
            src={mockPost.images[0]}
            alt="Post"
            className="max-h-full max-w-full object-contain"
          />
        )}

        {/* Action Buttons - Right Side */}
        <div className="fixed right-4 bottom-32 flex flex-col items-center gap-6 z-40">
          <button
            onClick={() => setIsLiked(!isLiked)}
            className="flex flex-col items-center gap-1 active:scale-95 transition-transform"
          >
            <div className="w-12 h-12 rounded-full bg-black/40 backdrop-blur-md flex items-center justify-center">
              <Heart
                className={`h-6 w-6 ${isLiked ? 'fill-red-500 text-red-500' : 'text-white'}`}
              />
            </div>
            <span className="text-white text-xs font-semibold">{mockPost.likes + (isLiked ? 1 : 0)}</span>
          </button>

          <button
            onClick={() => setShowComments(!showComments)}
            className="flex flex-col items-center gap-1 active:scale-95 transition-transform"
          >
            <div className="w-12 h-12 rounded-full bg-black/40 backdrop-blur-md flex items-center justify-center">
              <MessageCircle className="h-6 w-6 text-white" />
            </div>
            <span className="text-white text-xs font-semibold">{mockComments.length}</span>
          </button>

          <button className="flex flex-col items-center gap-1 active:scale-95 transition-transform">
            <div className="w-12 h-12 rounded-full bg-black/40 backdrop-blur-md flex items-center justify-center">
              <Share2 className="h-6 w-6 text-white" />
            </div>
          </button>
        </div>

        {/* Caption Overlay - Bottom */}
        <div className="fixed bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/60 to-transparent p-4 pb-20">
          <div className="max-w-xl mx-auto">
            <p className="text-white text-sm mb-2">
              <span className="font-semibold mr-2">{mockPost.user.username}</span>
              {mockPost.content}
            </p>
            <p className="text-gray-400 text-xs">
              {formatDistanceToNow(new Date(mockPost.createdAt), { addSuffix: true })}
            </p>
          </div>
        </div>
      </div>

      {/* Comments Overlay */}
      {showComments && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" onClick={() => setShowComments(false)}>
          <div className="absolute bottom-0 left-0 right-0 bg-white rounded-t-3xl max-h-[70vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="sticky top-0 bg-white border-b border-gray-100 px-6 py-4 flex items-center justify-between">
              <h3 className="font-bold text-lg">Comments</h3>
              <button
                onClick={() => setShowComments(false)}
                className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-100"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="px-6 py-4 space-y-4">
              {mockComments.map((comment) => (
                <div key={comment.id} className="flex gap-3">
                  <ProfileAvatar
                    src={comment.author?.avatarUrl || getPlaceholderAvatar(comment.author?.handle || 'User', 'user')}
                    alt={comment.author?.handle || 'User'}
                    type="user"
                    size="sm"
                    fallbackText={(comment.author?.handle || 'U').slice(0, 2).toUpperCase()}
                  />
                  <div className="flex-1">
                    <div className="flex items-baseline gap-2 mb-1">
                      <span className="font-bold text-sm">{comment.author?.handle || 'User'}</span>
                      <span className="text-xs text-gray-500">
                        {formatDistanceToNow(new Date(comment.createdAt), { addSuffix: true })}
                      </span>
                    </div>
                    <p className="text-sm text-gray-900">{comment.text}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Comment Input */}
            <div className="sticky bottom-0 bg-white border-t border-gray-100 px-6 py-4">
              <div className="flex items-center gap-3">
                <ProfileAvatar
                  src={getPlaceholderAvatar('You', 'user')}
                  alt="You"
                  type="user"
                  size="sm"
                  fallbackText="YO"
                />
                <Input
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="Add a comment..."
                  className="flex-1 rounded-full border-gray-200"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendComment();
                    }
                  }}
                />
                <button
                  onClick={handleSendComment}
                  disabled={!comment.trim()}
                  className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 disabled:opacity-30 flex items-center justify-center active:scale-95 transition-all"
                >
                  <Send className="h-4 w-4 text-white" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
