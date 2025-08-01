import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import Link from "next/link";
import { Footprints, Trophy, Users, Heart, Activity } from "lucide-react";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Woof - Social Fitness for Pets & Owners",
  description: "Connect with pet owners, track activities, and build a healthier, happier life with your furry friends.",
  keywords: ["Woof", "pets", "fitness", "social", "walking", "activities", "dog", "cat"],
  authors: [{ name: "Woof Team" }],
  openGraph: {
    title: "Woof - Social Fitness for Pets & Owners",
    description: "Connect with pet owners, track activities, and build a healthier, happier life with your furry friends.",
    url: "https://Woof.com",
    siteName: "Woof",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Woof - Social Fitness for Pets & Owners",
    description: "Connect with pet owners, track activities, and build a healthier, happier life with your furry friends.",
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
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-50 text-gray-900`}
      >
        <div className="min-h-screen flex flex-col">
          {/* Navigation Header */}
          <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center h-16">
                <div className="flex items-center">
                  <Link href="/" className="flex items-center space-x-2">
                    <Footprints className="w-8 h-8 text-blue-600" />
                    <span className="text-xl font-bold text-gray-900">Woof</span>
                  </Link>
                </div>
                
                {/* Desktop Navigation */}
                <nav className="hidden md:flex items-center space-x-8">
                  <Link 
                    href="/" 
                    className="flex items-center space-x-1 text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    <Activity className="w-4 h-4" />
                    <span>Dashboard</span>
                  </Link>
                  <Link 
                    href="/pets" 
                    className="flex items-center space-x-1 text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    <Footprints className="w-4 h-4" />
                    <span>Pets</span>
                  </Link>
                  <Link 
                    href="/activities" 
                    className="flex items-center space-x-1 text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    <Activity className="w-4 h-4" />
                    <span>Activities</span>
                  </Link>
                  <Link 
                    href="/social" 
                    className="flex items-center space-x-1 text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    <Users className="w-4 h-4" />
                    <span>Social</span>
                  </Link>
                  <Link 
                    href="/leaderboard" 
                    className="flex items-center space-x-1 text-gray-600 hover:text-gray-900 transition-colors"
                  >
                    <Trophy className="w-4 h-4" />
                    <span>Leaderboard</span>
                  </Link>
                </nav>

                {/* User Menu */}
                <div className="flex items-center space-x-4">
                  <div className="hidden md:flex items-center space-x-2 text-sm">
                    <Trophy className="w-4 h-4 text-yellow-500" />
                    <span className="font-medium">2,450 pts</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center">
                      <span className="text-white text-sm font-medium">JD</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </header>

          {/* Mobile Bottom Navigation */}
          <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 z-50">
            <div className="grid grid-cols-5 h-16">
              <Link 
                href="/" 
                className="flex flex-col items-center justify-center text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Activity className="w-5 h-5" />
                <span className="text-xs mt-1">Home</span>
              </Link>
              <Link 
                href="/pets" 
                className="flex flex-col items-center justify-center text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Footprints className="w-5 h-5" />
                <span className="text-xs mt-1">Pets</span>
              </Link>
              <Link 
                href="/activities" 
                className="flex flex-col items-center justify-center text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Activity className="w-5 h-5" />
                <span className="text-xs mt-1">Activities</span>
              </Link>
              <Link 
                href="/social" 
                className="flex flex-col items-center justify-center text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Users className="w-5 h-5" />
                <span className="text-xs mt-1">Social</span>
              </Link>
              <Link 
                href="/leaderboard" 
                className="flex flex-col items-center justify-center text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Trophy className="w-5 h-5" />
                <span className="text-xs mt-1">Ranks</span>
              </Link>
            </div>
          </nav>

          {/* Main Content */}
          <main className="flex-1 pb-16 md:pb-0">
            {children}
          </main>
        </div>
        <Toaster />
      </body>
    </html>
  );
}
