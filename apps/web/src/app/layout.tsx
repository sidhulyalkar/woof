import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { Suspense } from "react"
import "./globals.css"
import { Toaster } from "@/components/ui/toaster"
import { ServiceWorkerRegister } from "@/components/service-worker-register"
import { Providers } from "@/components/providers"
import { AuthGuard } from "@/components/auth-guard"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
})

export const metadata: Metadata = {
  title: "PetPath - Connect with Compatible Pet Owners",
  description: "Find compatible pet owners, track activities, and join community events",
  generator: "v0.app",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "PetPath",
  },
  formatDetection: {
    telephone: false,
  },
  themeColor: "#FF6B6B",
  viewport: {
    width: "device-width",
    initialScale: 1,
    maximumScale: 1,
  },
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
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
      </head>
      <body className={`font-sans ${inter.variable} ${GeistMono.variable} antialiased`}>
        <Providers>
          <AuthGuard>
            <Suspense fallback={null}>{children}</Suspense>
          </AuthGuard>
        </Providers>
        <Analytics />
        <Toaster />
        <ServiceWorkerRegister />
      </body>
    </html>
  )
}
