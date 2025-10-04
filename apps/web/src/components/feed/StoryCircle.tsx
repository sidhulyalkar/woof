'use client';

import { EmojiAvatar, getEmojiForId, getVariantForId } from '@/components/ui/EmojiAvatar';
import { Plus } from 'lucide-react';

interface StoryCircleProps {
  id?: string;
  username: string;
  avatar?: string;
  hasStory?: boolean;
  isOwn?: boolean;
  onClick?: () => void;
}

export function StoryCircle({ id, username, hasStory = false, isOwn = false, onClick }: StoryCircleProps) {
  const userId = id || username; // Use id if available, fallback to username

  return (
    <button
      onClick={onClick}
      className="flex flex-col items-center gap-2 min-w-[72px] group"
    >
      <div className="relative">
        <div
          className={`rounded-full p-[3px] transition-all duration-300 ${
            hasStory
              ? 'bg-gradient-to-tr from-yellow-400 via-pink-500 to-purple-500 group-hover:scale-105'
              : 'bg-gradient-to-br from-gray-200 to-gray-300 group-hover:scale-105'
          }`}
        >
          <div className="bg-white rounded-full p-[3px]">
            <EmojiAvatar
              emoji={getEmojiForId(userId, 'user')}
              variant={getVariantForId(userId)}
              size="lg"
              className="ring-0"
            />
          </div>
        </div>

        {isOwn && (
          <div className="absolute -bottom-1 -right-1 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full p-1.5 border-[3px] border-white shadow-lg transition-transform duration-300 group-hover:scale-110">
            <Plus className="h-3 w-3 text-white stroke-[3]" />
          </div>
        )}
      </div>

      <span className="text-xs font-semibold truncate max-w-[72px] text-gray-900">
        {isOwn ? 'Your story' : username}
      </span>
    </button>
  );
}
