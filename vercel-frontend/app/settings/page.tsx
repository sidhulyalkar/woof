"use client"

import { useState } from "react"
import { SettingsIcon, User, Bell, Lock, Palette, Globe, HelpCircle, LogOut, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { BottomNav } from "@/components/bottom-nav"

export default function SettingsPage() {
  const [notifications, setNotifications] = useState({
    matches: true,
    messages: true,
    events: true,
    health: false,
  })

  const [privacy, setPrivacy] = useState({
    showLocation: true,
    showActivity: true,
    allowMessages: true,
  })

  const [darkMode, setDarkMode] = useState(false)

  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <div className="sticky top-0 z-10 border-b border-border/40 bg-background/80 backdrop-blur-xl">
        <div className="flex items-center gap-3 px-4 py-4">
          <SettingsIcon className="h-6 w-6 text-accent" />
          <h1 className="text-xl font-bold">Settings</h1>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* Profile Section */}
        <Card className="glass p-4">
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16 border-2 border-border">
              <AvatarImage src="/user-avatar.jpg" alt="John Doe" />
              <AvatarFallback>JD</AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <p className="font-semibold">John Doe</p>
              <p className="text-sm text-muted-foreground">john.doe@example.com</p>
            </div>
            <Button variant="ghost" size="icon">
              <ChevronRight className="h-5 w-5" />
            </Button>
          </div>
        </Card>

        {/* Account Settings */}
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-muted-foreground px-2">ACCOUNT</h2>
          <Card className="glass divide-y divide-border/40">
            <button className="flex w-full items-center gap-3 p-4 text-left hover:bg-accent/5 transition-colors">
              <User className="h-5 w-5 text-muted-foreground" />
              <span className="flex-1">Edit Profile</span>
              <ChevronRight className="h-5 w-5 text-muted-foreground" />
            </button>
            <button className="flex w-full items-center gap-3 p-4 text-left hover:bg-accent/5 transition-colors">
              <Lock className="h-5 w-5 text-muted-foreground" />
              <span className="flex-1">Privacy & Security</span>
              <ChevronRight className="h-5 w-5 text-muted-foreground" />
            </button>
          </Card>
        </div>

        {/* Notifications */}
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-muted-foreground px-2">NOTIFICATIONS</h2>
          <Card className="glass divide-y divide-border/40">
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <Bell className="h-5 w-5 text-muted-foreground" />
                <span>New Matches</span>
              </div>
              <Switch
                checked={notifications.matches}
                onCheckedChange={(v) => setNotifications({ ...notifications, matches: v })}
              />
            </div>
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <Bell className="h-5 w-5 text-muted-foreground" />
                <span>Messages</span>
              </div>
              <Switch
                checked={notifications.messages}
                onCheckedChange={(v) => setNotifications({ ...notifications, messages: v })}
              />
            </div>
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <Bell className="h-5 w-5 text-muted-foreground" />
                <span>Events</span>
              </div>
              <Switch
                checked={notifications.events}
                onCheckedChange={(v) => setNotifications({ ...notifications, events: v })}
              />
            </div>
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <Bell className="h-5 w-5 text-muted-foreground" />
                <span>Health Reminders</span>
              </div>
              <Switch
                checked={notifications.health}
                onCheckedChange={(v) => setNotifications({ ...notifications, health: v })}
              />
            </div>
          </Card>
        </div>

        {/* Privacy */}
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-muted-foreground px-2">PRIVACY</h2>
          <Card className="glass divide-y divide-border/40">
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <Globe className="h-5 w-5 text-muted-foreground" />
                <span>Show Location</span>
              </div>
              <Switch
                checked={privacy.showLocation}
                onCheckedChange={(v) => setPrivacy({ ...privacy, showLocation: v })}
              />
            </div>
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <Globe className="h-5 w-5 text-muted-foreground" />
                <span>Show Activity Status</span>
              </div>
              <Switch
                checked={privacy.showActivity}
                onCheckedChange={(v) => setPrivacy({ ...privacy, showActivity: v })}
              />
            </div>
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <Globe className="h-5 w-5 text-muted-foreground" />
                <span>Allow Messages from Anyone</span>
              </div>
              <Switch
                checked={privacy.allowMessages}
                onCheckedChange={(v) => setPrivacy({ ...privacy, allowMessages: v })}
              />
            </div>
          </Card>
        </div>

        {/* Appearance */}
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-muted-foreground px-2">APPEARANCE</h2>
          <Card className="glass">
            <div className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <Palette className="h-5 w-5 text-muted-foreground" />
                <span>Dark Mode</span>
              </div>
              <Switch checked={darkMode} onCheckedChange={setDarkMode} />
            </div>
          </Card>
        </div>

        {/* Support */}
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-muted-foreground px-2">SUPPORT</h2>
          <Card className="glass divide-y divide-border/40">
            <button className="flex w-full items-center gap-3 p-4 text-left hover:bg-accent/5 transition-colors">
              <HelpCircle className="h-5 w-5 text-muted-foreground" />
              <span className="flex-1">Help Center</span>
              <ChevronRight className="h-5 w-5 text-muted-foreground" />
            </button>
            <button className="flex w-full items-center gap-3 p-4 text-left hover:bg-accent/5 transition-colors">
              <HelpCircle className="h-5 w-5 text-muted-foreground" />
              <span className="flex-1">Terms of Service</span>
              <ChevronRight className="h-5 w-5 text-muted-foreground" />
            </button>
            <button className="flex w-full items-center gap-3 p-4 text-left hover:bg-accent/5 transition-colors">
              <HelpCircle className="h-5 w-5 text-muted-foreground" />
              <span className="flex-1">Privacy Policy</span>
              <ChevronRight className="h-5 w-5 text-muted-foreground" />
            </button>
          </Card>
        </div>

        {/* Logout */}
        <Button variant="outline" className="w-full gap-2 bg-transparent text-destructive hover:bg-destructive/10">
          <LogOut className="h-5 w-5" />
          Log Out
        </Button>

        {/* App Version */}
        <p className="text-center text-xs text-muted-foreground">PetPath v1.0.0</p>
      </div>

      <BottomNav />
    </div>
  )
}
