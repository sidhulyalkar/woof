'use client';

import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { cn } from '@/lib/utils';
import { User, Dog } from 'lucide-react';

interface ProfileAvatarProps {
  src?: string | null;
  alt?: string;
  type?: 'user' | 'pet';
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';
  className?: string;
  fallbackText?: string;
}

const sizeClasses = {
  xs: 'w-6 h-6 text-[10px]',
  sm: 'w-8 h-8 text-xs',
  md: 'w-10 h-10 text-sm',
  lg: 'w-12 h-12 text-base',
  xl: 'w-16 h-16 text-lg',
  '2xl': 'w-20 h-20 text-xl',
};

const iconSizes = {
  xs: 12,
  sm: 14,
  md: 16,
  lg: 18,
  xl: 24,
  '2xl': 32,
};

export function ProfileAvatar({
  src,
  alt = 'Profile',
  type = 'user',
  size = 'md',
  className,
  fallbackText,
}: ProfileAvatarProps) {
  const IconComponent = type === 'pet' ? Dog : User;
  const iconSize = iconSizes[size];

  return (
    <Avatar className={cn(sizeClasses[size], 'ring-2 ring-white shadow-sm', className)}>
      <AvatarImage src={src || undefined} alt={alt} className="object-cover" />
      <AvatarFallback className="bg-gradient-to-br from-gray-100 to-gray-200 text-gray-600">
        {fallbackText ? (
          <span className="font-semibold">{fallbackText}</span>
        ) : (
          <IconComponent size={iconSize} className="text-gray-500" />
        )}
      </AvatarFallback>
    </Avatar>
  );
}

// Helper function to generate placeholder avatar URL with initials
export function getPlaceholderAvatar(name: string, type: 'user' | 'pet' = 'user'): string {
  const colors = type === 'user'
    ? ['3B82F6', '8B5CF6', 'EC4899', '10B981', 'F59E0B'] // User colors: blue, purple, pink, green, orange
    : ['F59E0B', 'EF4444', '8B5CF6', '06B6D4', '10B981']; // Pet colors: orange, red, purple, cyan, green

  const hash = name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const color = colors[hash % colors.length];
  const initials = name
    .split(' ')
    .map(word => word[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);

  return `https://ui-avatars.com/api/?name=${encodeURIComponent(initials)}&background=${color}&color=fff&size=128&bold=true`;
}
