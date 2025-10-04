'use client';

import { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { EmojiAvatar, getEmojiForId, getVariantForId } from '@/components/ui/EmojiAvatar';
import { useSessionStore } from '@/store/session';
import { formatDistanceToNow } from 'date-fns';

export interface Comment {
  id: string;
  postId: string;
  authorUserId: string;
  text: string;
  createdAt: string;
  author?: {
    id: string;
    username: string;
    avatar?: string;
  };
}

interface CommentSectionProps {
  postId: string;
  comments: Comment[];
  isLoading?: boolean;
  onAddComment: (text: string) => Promise<void>;
  onDeleteComment?: (commentId: string) => Promise<void>;
}

export function CommentSection({
  postId,
  comments,
  isLoading,
  onAddComment,
  onDeleteComment
}: CommentSectionProps) {
  const { user } = useSessionStore();
  const [newComment, setNewComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newComment.trim() || isSubmitting) return;

    setIsSubmitting(true);
    try {
      await onAddComment(newComment.trim());
      setNewComment('');
    } catch (error) {
      console.error('Failed to add comment:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async (commentId: string) => {
    if (onDeleteComment) {
      try {
        await onDeleteComment(commentId);
      } catch (error) {
        console.error('Failed to delete comment:', error);
      }
    }
  };

  return (
    <div className="space-y-3">
      {/* Comments List - Figma Style: Compact spacing */}
      {isLoading ? (
        <div className="flex items-center justify-center py-3">
          <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
        </div>
      ) : comments.length > 0 ? (
        <div className="space-y-3">
          {comments.map((comment) => (
            <div key={comment.id} className="flex gap-3 group">
              <EmojiAvatar
                emoji={getEmojiForId(comment.authorUserId, 'user')}
                variant={getVariantForId(comment.authorUserId)}
                size="sm"
              />

              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-2 mb-1">
                  <span className="font-bold text-sm text-gray-900">
                    {comment.author?.username || 'Anonymous'}
                  </span>
                  <span className="text-xs text-gray-500 font-medium">
                    {formatDistanceToNow(new Date(comment.createdAt), { addSuffix: true })}
                  </span>
                  {user?.id === comment.authorUserId && onDeleteComment && (
                    <button
                      onClick={() => handleDelete(comment.id)}
                      className="ml-auto text-xs font-medium text-gray-500 hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100"
                    >
                      Delete
                    </button>
                  )}
                </div>
                <p className="text-[15px] leading-5 text-gray-700">{comment.text}</p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-gray-500 text-center py-2">
          No comments yet
        </p>
      )}

      {/* Premium Comment Input */}
      <form onSubmit={handleSubmit} className="flex gap-3 pt-3 border-t border-gray-100">
        <EmojiAvatar
          emoji={getEmojiForId(user?.id || '', 'user')}
          variant={getVariantForId(user?.id || '')}
          size="sm"
        />

        <div className="flex-1 flex gap-2">
          <input
            type="text"
            value={newComment}
            onChange={(e) => setNewComment(e.target.value)}
            placeholder="Add a comment..."
            className="flex-1 px-4 py-2 bg-gray-50 border border-gray-200 rounded-full text-[15px] text-gray-900 placeholder:text-gray-500 focus:outline-none focus:border-blue-400 focus:bg-white transition-all"
            disabled={isSubmitting}
          />
          <button
            type="submit"
            disabled={!newComment.trim() || isSubmitting}
            className="w-9 h-9 flex items-center justify-center rounded-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 disabled:from-gray-300 disabled:to-gray-300 disabled:cursor-not-allowed shadow-md hover:shadow-lg active:scale-95 transition-all duration-200"
          >
            {isSubmitting ? (
              <Loader2 className="h-4 w-4 animate-spin text-white" />
            ) : (
              <Send className="h-4 w-4 text-white" strokeWidth={2.5} />
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
