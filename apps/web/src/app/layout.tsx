import type { Metadata } from 'next';
import { Space_Grotesk, Inter } from 'next/font/google';
import './globals.css';
import { ThemeProvider } from '@/components/theme-provider';
import { Toaster } from 'sonner';

const spaceGrotesk = Space_Grotesk({
  variable: '--font-space-grotesk',
  subsets: ['latin'],
  display: 'swap',
});

const inter = Inter({
  variable: '--font-inter',
  subsets: ['latin'],
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'PetPath - Social Fitness for Pets & Owners',
  description: 'Connect with pet owners, track activities, and build a healthier, happier life with your furry friends.',
  keywords: ['PetPath', 'pets', 'fitness', 'social', 'walking', 'activities', 'dog', 'cat'],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${spaceGrotesk.variable} ${inter.variable} antialiased`}
        style={{
          fontFamily: 'var(--font-body)',
        }}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          <div
            className="min-h-screen w-full flex items-center justify-center"
            style={{ backgroundColor: '#0A0F1E' }}
          >
            <div
              className="w-full min-h-screen relative"
              style={{
                maxWidth: '448px',
                backgroundColor: 'var(--background)',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
              }}
            >
              {children}
            </div>
          </div>
          <Toaster position="top-center" />
        </ThemeProvider>
      </body>
    </html>
  );
}
