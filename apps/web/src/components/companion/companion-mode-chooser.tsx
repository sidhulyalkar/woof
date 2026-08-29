'use client';

import { HeartHandshake, Home, PawPrint } from 'lucide-react';
import type { CompanionMode } from '@/lib/api/companion';

const options: Array<{
  mode: CompanionMode;
  title: string;
  description: string;
  icon: typeof PawPrint;
}> = [
  {
    mode: 'PET_GUARDIAN',
    title: 'I live with a dog',
    description: 'Use pet-specific Today, Story, Coach, Adventure, and relationship learning.',
    icon: PawPrint,
  },
  {
    mode: 'ANIMAL_ALLY',
    title: 'I want to learn and help',
    description:
      'Practice human skills, join the community, and prepare without inventing a pet profile.',
    icon: HeartHandshake,
  },
  {
    mode: 'FOSTER_CAREGIVER',
    title: 'I foster or plan to foster',
    description:
      'Build caregiver skills and practical readiness while pet authority stays relationship-specific.',
    icon: Home,
  },
];

export function CompanionModeChooser({
  onSelect,
  disabled = false,
  compact = false,
}: {
  onSelect: (mode: CompanionMode) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  return (
    <div className={compact ? 'grid gap-2' : 'grid gap-3'}>
      {options.map((option) => {
        const Icon = option.icon;
        return (
          <button
            key={option.mode}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(option.mode)}
            className="group rounded-2xl border border-border/70 bg-card/70 p-4 text-left transition hover:border-primary/35 hover:bg-primary/[0.04] disabled:cursor-not-allowed disabled:opacity-60"
          >
            <div className="flex items-start gap-3">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <Icon className="h-5 w-5" aria-hidden="true" />
              </span>
              <div>
                <p className="font-bold tracking-tight">{option.title}</p>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  {option.description}
                </p>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
