'use client';

import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { Plus } from 'lucide-react';

interface StoryCircleProps {
  id?: string;
  username: string;
  avatar?: string;
  hasStory?: boolean;
  isOwn?: boolean;
  onClick?: () => void;
}

export function StoryCircle({ username, avatar, hasStory = false, isOwn = false, onClick }: StoryCircleProps) {
  return (
    <button
      onClick={onClick}
      className="flex flex-col items-center gap-1.5 min-w-[68px] group"
    >
      <div className="relative">
        <div
          className={`rounded-full p-[2px] transition-all duration-200 ${
            hasStory
              ? 'bg-gradient-to-tr from-yellow-400 via-pink-500 to-purple-500 group-hover:scale-105'
              : 'bg-border/10 group-hover:bg-border/20'
          }`}
        >
          <div className="bg-background rounded-full p-[2px]">
            <Avatar className="h-[60px] w-[60px] ring-2 ring-background">
              <AvatarImage src={avatar} />
              <AvatarFallback className="bg-gradient-to-br from-purple-500 to-pink-500 text-white text-sm font-medium">
                {username[0].toUpperCase()}
              </AvatarFallback>
            </Avatar>
          </div>
        </div>

        {isOwn && (
          <div className="absolute -bottom-0.5 -right-0.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full p-1 border-[2.5px] border-background shadow-lg transition-transform duration-200 group-hover:scale-110">
            <Plus className="h-3 w-3 text-white stroke-[2.5px]" />
          </div>
        )}
      </div>

      <span className="text-[11px] font-medium truncate max-w-[68px] text-foreground/80">
        {isOwn ? 'Your story' : username}
      </span>
    </button>
  );
}
