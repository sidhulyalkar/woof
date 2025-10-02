'use client';

import { Heart } from 'lucide-react';

interface HappinessScoreProps {
  score: number;
}

export function HappinessScore({ score }: HappinessScoreProps) {
  const circumference = 2 * Math.PI * 36;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className="mt-6 bg-white/10 rounded-xl p-4 backdrop-blur-sm border border-white/20">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-accent-100">Weekly Happiness Score</p>
          <p className="text-3xl font-heading font-bold">{score}%</p>
        </div>
        <div className="relative w-20 h-20">
          <svg className="w-full h-full transform -rotate-90">
            <circle
              cx="40"
              cy="40"
              r="36"
              stroke="rgba(255,255,255,0.2)"
              strokeWidth="6"
              fill="none"
            />
            <circle
              cx="40"
              cy="40"
              r="36"
              stroke="white"
              strokeWidth="6"
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-1000"
            />
          </svg>
          <Heart className="absolute inset-0 m-auto w-8 h-8 text-white fill-white" />
        </div>
      </div>
    </div>
  );
}
