import React from 'react';
import { cn } from '@/lib/utils';

interface EmojiAvatarProps {
  emoji?: string;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';
  variant?: 'blue' | 'purple' | 'pink' | 'green' | 'orange' | 'gradient';
  className?: string;
}

const sizeClasses = {
  xs: 'w-8 h-8 text-base',
  sm: 'w-10 h-10 text-lg',
  md: 'w-12 h-12 text-xl',
  lg: 'w-14 h-14 text-2xl',
  xl: 'w-16 h-16 text-3xl',
  '2xl': 'w-20 h-20 text-4xl',
};

const variantClasses = {
  blue: 'bg-gradient-to-br from-blue-400 via-blue-500 to-indigo-600',
  purple: 'bg-gradient-to-br from-purple-400 via-purple-500 to-pink-600',
  pink: 'bg-gradient-to-br from-pink-400 via-rose-500 to-red-600',
  green: 'bg-gradient-to-br from-emerald-400 via-green-500 to-teal-600',
  orange: 'bg-gradient-to-br from-orange-400 via-amber-500 to-yellow-600',
  gradient: 'bg-gradient-to-br from-violet-400 via-fuchsia-500 to-pink-600',
};

// Default emojis for fallback
const defaultEmojis = ['🐕', '🐈', '🐾', '🦴', '🎾', '🐶', '🐱', '🦮', '🐕‍🦺'];

export function EmojiAvatar({
  emoji,
  size = 'md',
  variant = 'gradient',
  className
}: EmojiAvatarProps) {
  const displayEmoji = emoji || defaultEmojis[Math.floor(Math.random() * defaultEmojis.length)];

  return (
    <div
      className={cn(
        'relative flex items-center justify-center rounded-full',
        'backdrop-blur-xl bg-white/20 ring-2 ring-white/40',
        'shadow-lg hover:shadow-xl transition-all duration-300',
        'group cursor-pointer',
        variantClasses[variant],
        sizeClasses[size],
        className
      )}
    >
      {/* Glass effect overlay */}
      <div className="absolute inset-0 rounded-full bg-gradient-to-br from-white/30 to-transparent opacity-60" />

      {/* Emoji */}
      <span className="relative z-10 select-none filter drop-shadow-sm group-hover:scale-110 transition-transform duration-300">
        {displayEmoji}
      </span>

      {/* Hover glow effect */}
      <div className="absolute inset-0 rounded-full bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-md" />
    </div>
  );
}

// Preset emojis for users and pets
export const userEmojis = [
  '👤', '🧑', '👨', '👩', '🧔', '👨‍🦱', '👩‍🦱', '👨‍🦰', '👩‍🦰',
  '👨‍🦳', '👩‍🦳', '👨‍🦲', '👩‍🦲', '🧑‍🦱', '🧑‍🦰', '🧑‍🦳', '🧑‍🦲'
];

export const petEmojis = {
  dog: ['🐕', '🐶', '🦮', '🐕‍🦺', '🐩'],
  cat: ['🐈', '🐱', '🐈‍⬛'],
  other: ['🐾', '🦴', '🎾', '🦆', '🐇', '🐹', '🐦', '🐠', '🐢']
};

// Helper function to get a consistent emoji based on ID
export function getEmojiForId(id: string, type: 'user' | 'pet' = 'user'): string {
  const emojis = type === 'user' ? userEmojis : [...petEmojis.dog, ...petEmojis.cat, ...petEmojis.other];
  const hash = id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return emojis[hash % emojis.length];
}

// Helper function to get color variant based on ID
export function getVariantForId(id: string): 'blue' | 'purple' | 'pink' | 'green' | 'orange' | 'gradient' {
  const variants: Array<'blue' | 'purple' | 'pink' | 'green' | 'orange' | 'gradient'> = ['blue', 'purple', 'pink', 'green', 'orange', 'gradient'];
  const hash = id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return variants[hash % variants.length];
}
