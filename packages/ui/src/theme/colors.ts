/**
 * Woof Brand Colors - Galaxy Dark Theme
 *
 * Primary: Deep cosmic blue (#0B1C3D)
 * Secondary: Rich navy (#0F2A6B)
 * Accent: Bright blue (#6BA8FF)
 * Surface: Dark slate (#0E1220)
 */

export const colors = {
  // Brand Colors
  primary: {
    DEFAULT: '#0B1C3D',
    50: '#E8EDF5',
    100: '#C8D5E8',
    200: '#91ABD1',
    300: '#5A81BA',
    400: '#2357A3',
    500: '#0B1C3D',
    600: '#091631',
    700: '#071125',
    800: '#050B19',
    900: '#03060D',
  },
  secondary: {
    DEFAULT: '#0F2A6B',
    50: '#E9EEF8',
    100: '#CAD7EF',
    200: '#95AFDF',
    300: '#6087CF',
    400: '#2B5FBF',
    500: '#0F2A6B',
    600: '#0C2256',
    700: '#091941',
    800: '#06112C',
    900: '#030917',
  },
  accent: {
    DEFAULT: '#6BA8FF',
    50: '#FFFFFF',
    100: '#F5F9FF',
    200: '#D6E7FF',
    300: '#B7D5FF',
    400: '#98C4FF',
    500: '#6BA8FF',
    600: '#3D8CFF',
    700: '#0F70FF',
    800: '#0058D0',
    900: '#0044A2',
  },
  surface: {
    DEFAULT: '#0E1220',
    50: '#4A5469',
    100: '#3F4859',
    200: '#2A3141',
    300: '#1E2332',
    400: '#161A26',
    500: '#0E1220',
    600: '#0B0E1A',
    700: '#080A14',
    800: '#05060E',
    900: '#020308',
  },

  // Semantic Colors
  success: {
    DEFAULT: '#10B981',
    light: '#34D399',
    dark: '#059669',
  },
  warning: {
    DEFAULT: '#F59E0B',
    light: '#FBBF24',
    dark: '#D97706',
  },
  error: {
    DEFAULT: '#EF4444',
    light: '#F87171',
    dark: '#DC2626',
  },
  info: {
    DEFAULT: '#3B82F6',
    light: '#60A5FA',
    dark: '#2563EB',
  },

  // Neutral
  gray: {
    50: '#F9FAFB',
    100: '#F3F4F6',
    200: '#E5E7EB',
    300: '#D1D5DB',
    400: '#9CA3AF',
    500: '#6B7280',
    600: '#4B5563',
    700: '#374151',
    800: '#1F2937',
    900: '#111827',
    950: '#030712',
  },

  // Special
  white: '#FFFFFF',
  black: '#000000',
  transparent: 'transparent',
} as const;

export type ColorKey = keyof typeof colors;
