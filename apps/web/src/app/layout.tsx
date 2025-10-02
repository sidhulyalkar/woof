import type { Metadata } from 'next';
import { Space_Grotesk, Inter } from 'next/font/google';
import './globals.css';
import { ThemeProvider } from '@/components/theme-provider';
import { Providers } from '@/components/providers';
import { Header } from '@/components/layout/header';
import { MobileNav } from '@/components/layout/mobile-nav';
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
  title: 'Woof - Social Fitness for Pets & Owners',
  description: 'Connect with pet owners, track activities, and build a healthier, happier life with your furry friends.',
  keywords: ['Woof', 'pets', 'fitness', 'social', 'walking', 'activities', 'dog', 'cat'],
  authors: [{ name: 'Woof Team' }],
  openGraph: {
    title: 'Woof - Social Fitness for Pets & Owners',
    description: 'Connect with pet owners, track activities, and build a healthier, happier life with your furry friends.',
    url: 'https://woof.app',
    siteName: 'Woof',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Woof - Social Fitness for Pets & Owners',
    description: 'Connect with pet owners, track activities, and build a healthier, happier life with your furry friends.',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${spaceGrotesk.variable} ${inter.variable} font-sans antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          <Providers>
            <div className="min-h-screen flex flex-col bg-surface dark:bg-surface neuron-bg">
              <Header />
              <main className="flex-1 pb-16 md:pb-0">
                {children}
              </main>
              <MobileNav />
            </div>
            <Toaster position="top-center" />
          </Providers>
        </ThemeProvider>
      </body>
    </html>
  );
}
