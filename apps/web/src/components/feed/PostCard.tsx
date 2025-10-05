'use client';

import { useState } from 'react';
import { Heart, MessageCircle, Share2, MoreHorizontal } from 'lucide-react';
import { ProfileAvatar, getPlaceholderAvatar } from '@/components/ui/ProfileAvatar';
import { useLikePost, useComments, useCreateComment, useDeleteComment } from '@/lib/api/hooks';
import { useUIStore } from '@/store/ui';
import { formatDistanceToNow } from 'date-fns';
import type { Post } from '@/lib/api/hooks';
import { CommentSection } from './CommentSection';

interface PostCardProps {
  post: Post;
}

export function PostCard({ post }: PostCardProps) {
  const { showToast } = useUIStore();
  const [isLiked, setIsLiked] = useState(false);
  const [likesCount, setLikesCount] = useState(post.likes);
  const [showComments, setShowComments] = useState(false);

  const { data: comments = [], isLoading: commentsLoading } = useComments(post.id, {
    enabled: showComments,
  });
  const createCommentMutation = useCreateComment();
  const deleteCommentMutation = useDeleteComment();

  const likePostMutation = useLikePost({
    onSuccess: () => {
      setIsLiked(!isLiked);
      setLikesCount((prev) => (isLiked ? prev - 1 : prev + 1));
    },
    onError: () => {
      showToast({ message: 'Failed to like post', type: 'error' });
    },
  });

  const handleLike = () => {
    setIsLiked(!isLiked);
    setLikesCount((prev) => (isLiked ? prev - 1 : prev + 1));
    likePostMutation.mutate(post.id);
  };

  const handleComment = () => {
    setShowComments(!showComments);
  };

  const handleShare = () => {
    showToast({ message: 'Share coming soon!', type: 'info' });
  };

  const handleAddComment = async (text: string) => {
    await createCommentMutation.mutateAsync({ postId: post.id, text });
  };

  const handleDeleteComment = async (commentId: string) => {
    await deleteCommentMutation.mutateAsync({ commentId, postId: post.id });
  };

  return (
    <article className="bg-transparent">
      {/* Premium Instagram-Style Header */}
      <div className="flex items-center justify-between px-5 py-4">
        <div className="flex items-center gap-3">
          <ProfileAvatar
            src={post.user.avatar || getPlaceholderAvatar(post.user.username, 'user')}
            alt={post.user.username}
            type="user"
            size="lg"
            fallbackText={post.user.username.slice(0, 2).toUpperCase()}
          />

          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="font-bold text-[15px] text-gray-900">{post.user.username}</span>
              {post.pet && (
                <>
                  <span className="text-gray-300">·</span>
                  <span className="text-sm font-medium text-gray-600">{post.pet.name}</span>
                </>
              )}
            </div>
            <time className="text-xs text-gray-500 font-medium">
              {formatDistanceToNow(new Date(post.createdAt), { addSuffix: true })}
            </time>
          </div>
        </div>

        <button className="w-9 h-9 flex items-center justify-center rounded-full hover:bg-gray-100 active:scale-95 transition-all duration-200">
          <MoreHorizontal className="h-5 w-5 text-gray-700" />
        </button>
      </div>

      {/* Content - Premium Style: More spacious */}
      {post.content && (
        <div className="px-5 pb-3">
          <p className="text-[15px] leading-6 text-gray-900">
            {post.content}
          </p>
        </div>
      )}

      {/* Images - Figma Style: Edge-to-edge */}
      {post.images && post.images.length > 0 && (
        <div className={`grid gap-0.5 ${
          post.images.length === 1 ? 'grid-cols-1' :
          post.images.length === 2 ? 'grid-cols-2' :
          post.images.length === 3 ? 'grid-cols-2' : 'grid-cols-2'
        }`}>
          {post.images.slice(0, 4).map((image, index) => (
            <div
              key={index}
              className={`relative bg-gray-100 overflow-hidden ${
                post.images!.length === 1 ? 'aspect-video' : 'aspect-square'
              } ${
                post.images!.length === 3 && index === 0 ? 'col-span-2 aspect-video' : ''
              }`}
            >
              <img
                src={image}
                alt={`Post image ${index + 1}`}
                className="w-full h-full object-cover"
              />
              {index === 3 && post.images!.length > 4 && (
                <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                  <span className="text-white text-base font-medium">
                    +{post.images!.length - 4}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Premium Instagram-Style Actions */}
      <div className="flex items-center gap-5 px-5 py-3">
        <button
          onClick={handleLike}
          className={`flex items-center gap-2 group ${
            isLiked ? 'text-red-500' : 'text-gray-700'
          }`}
        >
          <Heart
            className={`h-6 w-6 transition-all duration-300 ${
              isLiked ? 'fill-current scale-110' : 'group-hover:scale-110'
            }`}
            strokeWidth={2}
          />
          <span className="text-sm font-semibold">{likesCount}</span>
        </button>

        <button
          onClick={handleComment}
          className="flex items-center gap-2 text-gray-700 group"
        >
          <MessageCircle className="h-6 w-6 group-hover:scale-110 transition-all duration-300" strokeWidth={2} />
          <span className="text-sm font-semibold">{post.comments}</span>
        </button>

        <button
          onClick={handleShare}
          className="flex items-center gap-2 text-gray-700 ml-auto group"
        >
          <Share2 className="h-6 w-6 group-hover:scale-110 transition-all duration-300" strokeWidth={2} />
        </button>
      </div>

      {/* Likes Count */}
      {likesCount > 0 && (
        <div className="px-5 pb-2">
          <p className="text-sm font-semibold text-gray-900">{likesCount} likes</p>
        </div>
      )}

      {/* Comments Section */}
      {showComments && (
        <div className="px-5 pb-4 pt-2 border-t border-gray-100">
          <CommentSection
            postId={post.id}
            comments={comments.map(c => ({
              ...c,
              author: c.author ? {
                id: c.author.id,
                username: c.author.handle,
                avatar: c.author.avatarUrl,
              } : undefined
            }))}
            isLoading={commentsLoading}
            onAddComment={handleAddComment}
            onDeleteComment={handleDeleteComment}
          />
        </div>
      )}
    </article>
  );
}
