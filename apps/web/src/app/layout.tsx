import type React from 'react';
import type { Metadata, Viewport } from 'next';
import { headers } from 'next/headers';
import { Inter } from 'next/font/google';
import { GeistMono } from 'geist/font/mono';
import { Analytics } from '@vercel/analytics/next';
import { Suspense } from 'react';
import './globals.css';
import { Toaster } from '@/components/ui/toaster';
import { Providers } from '@/components/providers';
import { AuthGuard } from '@/components/auth-guard';
import { ErrorBoundary } from '@/components/error-boundary';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const productDescription =
  'Woof helps you choose one useful thing to do with your dog, notice how it went, and make the next shared moment easier.';

export const metadata: Metadata = {
  title: {
    default: 'Woof · A better day with your dog',
    template: '%s · Woof',
  },
  description: productDescription,
  applicationName: 'Woof',
  keywords: ['dog companion', 'dog activities', 'dog training', 'dog care', 'dog relationship'],
  openGraph: {
    type: 'website',
    siteName: 'Woof',
    title: 'Woof · A better day with your dog',
    description: productDescription,
  },
  twitter: {
    card: 'summary',
    title: 'Woof · A better day with your dog',
    description: productDescription,
  },
  formatDetection: {
    telephone: false,
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f8f6f1' },
    { media: '(prefers-color-scheme: dark)', color: '#0d1117' },
  ],
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // A nonce CSP is request-specific. Reading request headers at the root keeps
  // every App Router page dynamically rendered so Next can apply the middleware
  // nonce to framework, page, and inline runtime scripts during SSR.
  await headers();

  return (
    <html lang="en">
      <body className={`font-sans ${inter.variable} ${GeistMono.variable} antialiased`}>
        <a href="#main-content" className="skip-link">
          Skip to content
        </a>
        <ErrorBoundary>
          <Providers>
            <AuthGuard>
              <Suspense fallback={null}>{children}</Suspense>
            </AuthGuard>
          </Providers>
          <Analytics />
          <Toaster />
        </ErrorBoundary>
      </body>
    </html>
  );
}
