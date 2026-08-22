import type { QuizQuestion } from '@/types/quiz';

export const QUIZ_SECTIONS = [
  {
    id: 'schedule',
    title: 'Schedule',
    icon: 'calendar',
    description: 'When shared routines usually fit into real life.',
  },
  {
    id: 'activity',
    title: 'Activity Level',
    icon: 'activity',
    description: 'How much structured activity feels realistic for your household.',
  },
  {
    id: 'interests',
    title: 'Interests',
    icon: 'heart',
    description: 'The kinds of experiences you already enjoy sharing with your pet.',
  },
] as const;

export const QUIZ_QUESTIONS: QuizQuestion[] = [
  {
    id: 'preferred_times',
    sectionId: 'schedule',
    question: 'When do shared activities usually fit best?',
    type: 'multiple_select',
    required: true,
    options: [
      { id: 'early', value: 'early_morning', label: 'Early mornings (5–8 AM)' },
      { id: 'midday', value: 'midday', label: 'Mid-day (10 AM–2 PM)' },
      { id: 'evening', value: 'evening', label: 'Evenings (5–8 PM)' },
      { id: 'flexible', value: 'flexible', label: 'Flexible throughout the day' },
    ],
  },
  {
    id: 'walk_frequency',
    sectionId: 'activity',
    question: 'How active is a typical day with your pet?',
    description: 'Choose a rough level. This is context, not a score of good ownership.',
    type: 'scale',
    required: true,
    scaleRange: {
      min: 1,
      max: 5,
      minLabel: 'Mostly restful',
      maxLabel: 'Very active',
    },
  },
  {
    id: 'preferred_activities',
    sectionId: 'interests',
    question: 'Which activities do you both tend to enjoy?',
    type: 'multiple_select',
    required: false,
    allowCustom: true,
    options: [
      { id: 'explore', value: 'exploring', label: 'Hiking and outdoor exploring' },
      { id: 'sniff', value: 'sniffing', label: 'Sniffing and scent games' },
      { id: 'training', value: 'training', label: 'Training and skill games' },
      { id: 'water', value: 'water', label: 'Beach or water activities' },
    ],
  },
];

// Keep backward compatibility for older imports while maintaining one source of truth.
export const quizQuestions = QUIZ_QUESTIONS;
