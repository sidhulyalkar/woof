'use client';

import { Heart, MessageCircle, Share2, MapPin, Camera } from 'lucide-react';
import { formatRelativeTime, getInitials } from '@/lib/utils';
import { useState } from 'react';
import { socialApi } from '@/lib/api';

interface Post {
  id: string;
  content: string;
  imageUrl?: string;
  location?: string;
  createdAt: string;
  likesCount: number;
  commentsCount: number;
  isLiked: boolean;
  user: {
    id: string;
    handle: string;
    avatarUrl?: string;
  };
  pet: {
    id: string;
    name: string;
    species: string;
  };
}

interface SocialFeedProps {
  posts: Post[];
  isLoading: boolean;
}

export function SocialFeed({ posts, isLoading }: SocialFeedProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-heading font-bold text-gray-100">Social Feed</h2>
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-64 bg-surface/50 rounded-lg animate-pulse" />
        ))}
      </div>
    );
  }

  if (!posts || posts.length === 0) {
    return (
      <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg p-12 text-center">
        <Camera className="w-12 h-12 text-gray-500 mx-auto mb-4" />
        <h3 className="font-heading font-semibold text-gray-300 mb-2">No posts yet</h3>
        <p className="text-gray-400">Follow some pets to see their adventures!</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-heading font-bold text-gray-100">Social Feed</h2>
      {posts.map((post) => (
        <PostCard key={post.id} post={post} />
      ))}
    </div>
  );
}

function PostCard({ post }: { post: Post }) {
  const [isLiked, setIsLiked] = useState(post.isLiked);
  const [likesCount, setLikesCount] = useState(post.likesCount);

  const handleLike = async () => {
    try {
      if (isLiked) {
        await socialApi.unlikePost(post.id);
        setLikesCount(likesCount - 1);
      } else {
        await socialApi.likePost(post.id);
        setLikesCount(likesCount + 1);
      }
      setIsLiked(!isLiked);
    } catch (error) {
      console.error('Failed to like post:', error);
    }
  };

  return (
    <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg overflow-hidden hover:border-accent/30 transition-all">
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-accent to-accent-600 flex items-center justify-center text-white font-medium text-sm">
              {post.user.avatarUrl ? (
                <img src={post.user.avatarUrl} alt={post.user.handle} className="w-full h-full rounded-full object-cover" />
              ) : (
                <span>{getInitials(post.user.handle)}</span>
              )}
            </div>
            <div>
              <p className="font-medium text-gray-100">{post.user.handle}</p>
              <p className="text-sm text-gray-400">
                with {post.pet.name} • {formatRelativeTime(new Date(post.createdAt))}
              </p>
            </div>
          </div>
          {post.location && (
            <div className="flex items-center text-xs text-gray-400">
              <MapPin className="w-3 h-3 mr-1" />
              {post.location}
            </div>
          )}
        </div>

        <p className="text-gray-200 mb-4">{post.content}</p>

        {post.imageUrl && (
          <div className="rounded-lg overflow-hidden mb-4">
            <img src={post.imageUrl} alt="Post" className="w-full h-64 object-cover" />
          </div>
        )}

        <div className="flex items-center justify-between pt-3 border-t border-primary/20">
          <div className="flex space-x-4">
            <button
              onClick={handleLike}
              className={`flex items-center space-x-1 text-sm transition-colors ${
                isLiked ? 'text-red-400' : 'text-gray-400 hover:text-red-400'
              }`}
            >
              <Heart className={`w-4 h-4 ${isLiked ? 'fill-red-400' : ''}`} />
              <span>{likesCount}</span>
            </button>
            <button className="flex items-center space-x-1 text-sm text-gray-400 hover:text-accent transition-colors">
              <MessageCircle className="w-4 h-4" />
              <span>{post.commentsCount}</span>
            </button>
            <button className="flex items-center space-x-1 text-sm text-gray-400 hover:text-accent transition-colors">
              <Share2 className="w-4 h-4" />
              <span>Share</span>
            </button>
          </div>
          <span className="px-2 py-1 rounded-full text-xs bg-accent/10 text-accent border border-accent/20">
            {post.pet.species}
          </span>
        </div>
      </div>
    </div>
  );
}
