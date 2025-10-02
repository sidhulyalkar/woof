import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: ['class'],
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    '../../packages/ui/src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Galaxy-dark theme colors
        primary: {
          DEFAULT: '#0B1C3D',
          50: '#E6EBF4',
          100: '#CCD7E9',
          200: '#99AFD3',
          300: '#6687BD',
          400: '#335FA7',
          500: '#0B1C3D',
          600: '#091631',
          700: '#071125',
          800: '#050B19',
          900: '#03060D',
        },
        secondary: {
          DEFAULT: '#0F2A6B',
          50: '#E7EDF9',
          100: '#CFDBF3',
          200: '#9FB7E7',
          300: '#6F93DB',
          400: '#3F6FCF',
          500: '#0F2A6B',
          600: '#0C2256',
          700: '#091941',
          800: '#06112B',
          900: '#030916',
        },
        accent: {
          DEFAULT: '#6BA8FF',
          50: '#FFFFFF',
          100: '#F0F6FF',
          200: '#D4E5FF',
          300: '#B8D4FF',
          400: '#9CC3FF',
          500: '#6BA8FF',
          600: '#3F8DFF',
          700: '#0C6BFF',
          800: '#0054D9',
          900: '#003FAD',
        },
        surface: {
          DEFAULT: '#0E1220',
          50: '#3A4A6D',
          100: '#334361',
          200: '#263449',
          300: '#1A2531',
          400: '#141B28',
          500: '#0E1220',
          600: '#0B0E19',
          700: '#080B12',
          800: '#05070B',
          900: '#020304',
        },
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', 'sans-serif'],
        heading: ['var(--font-space-grotesk)', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};

export default config;
