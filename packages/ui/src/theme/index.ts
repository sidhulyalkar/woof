export * from './colors';
export * from './typography';
export * from './spacing';

/**
 * Woof Brand Theme
 * Galaxy-dark aesthetic with glassmorphism
 */
export const theme = {
  brand: {
    name: 'Woof',
    tagline: 'Pet Social Fitness Platform',
    colors: {
      primary: '#0B1C3D',
      secondary: '#0F2A6B',
      accent: '#6BA8FF',
      surface: '#0E1220',
      onSurface: '#FFFFFF',
    },
  },
  motion: {
    spring: {
      type: 'spring',
      stiffness: 260,
      damping: 20,
    },
    springMedium: {
      type: 'spring',
      stiffness: 200,
      damping: 18,
    },
    ease: {
      type: 'tween',
      ease: 'easeInOut',
      duration: 0.3,
    },
  },
  glassmorphism: {
    background: 'rgba(14, 18, 32, 0.7)',
    backdropFilter: 'blur(16px) saturate(180%)',
    border: '1px solid rgba(107, 168, 255, 0.1)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.24)',
  },
} as const;
