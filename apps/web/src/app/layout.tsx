import type React from "react"
import type { Metadata, Viewport } from "next"
import { Inter } from "next/font/google"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { Suspense } from "react"
import "./globals.css"
import { Toaster } from "@/components/ui/toaster"
import { ServiceWorkerRegister } from "@/components/service-worker-register"
import { Providers } from "@/components/providers"
import { AuthGuard } from "@/components/auth-guard"
import { ErrorBoundary } from "@/components/error-boundary"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
})

export const metadata: Metadata = {
  title: {
    default: "Woof · Better dog friendships, offline",
    template: "%s · Woof",
  },
  description:
    "Discover compatible dogs nearby, coordinate meetups, track shared activity, and build better real-world pet friendships.",
  applicationName: "Woof",
  manifest: "/manifest.json",
  keywords: [
    "dog social app",
    "pet compatibility",
    "dog meetups",
    "pet activity tracking",
    "pet community",
  ],
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Woof",
  },
  formatDetection: {
    telephone: false,
  },
  robots: {
    index: true,
    follow: true,
  },
}

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f8f6f1" },
    { media: "(prefers-color-scheme: dark)", color: "#0d1117" },
  ],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="apple-touch-icon" href="/icon-192.jpg" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
      </head>
      <body className={`font-sans ${inter.variable} ${GeistMono.variable} antialiased`}>
        <a
          href="#main-content"
          className="skip-link"
        >
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
          <ServiceWorkerRegister />
        </ErrorBoundary>
      </body>
    </html>
  )
}
