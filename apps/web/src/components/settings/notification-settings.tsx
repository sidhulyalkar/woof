"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { usePushNotifications } from "@/hooks/use-push-notifications"
import { Bell, BellOff, AlertCircle, CheckCircle2 } from "lucide-react"

export function NotificationSettings() {
  const { isSupported, isSubscribed, isLoading, permission, subscribe, unsubscribe } =
    usePushNotifications()

  const handleToggle = async () => {
    if (isSubscribed) {
      await unsubscribe()
    } else {
      await subscribe()
    }
  }

  if (!isSupported) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BellOff className="h-5 w-5" />
            Push Notifications
          </CardTitle>
          <CardDescription>Stay updated with real-time alerts</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Push notifications are not supported in your browser. Please use a modern browser like
              Chrome, Firefox, or Safari.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bell className="h-5 w-5" />
          Push Notifications
        </CardTitle>
        <CardDescription>Stay updated with real-time alerts</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Permission Status */}
        {permission === "denied" && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Notification permission was denied. To enable notifications, please update your browser
              settings.
            </AlertDescription>
          </Alert>
        )}

        {permission === "granted" && isSubscribed && (
          <Alert>
            <CheckCircle2 className="h-4 w-4" />
            <AlertDescription>
              Push notifications are enabled. You'll receive real-time alerts for meetup suggestions,
              achievements, and events.
            </AlertDescription>
          </Alert>
        )}

        {/* Main Toggle */}
        <div className="flex items-center justify-between space-x-2">
          <div className="space-y-0.5">
            <Label htmlFor="push-notifications" className="text-base">
              Enable Push Notifications
            </Label>
            <p className="text-sm text-muted-foreground">
              Receive notifications even when the app is closed
            </p>
          </div>
          <Switch
            id="push-notifications"
            checked={isSubscribed}
            onCheckedChange={handleToggle}
            disabled={isLoading || permission === "denied"}
          />
        </div>

        {/* Notification Types */}
        {isSubscribed && (
          <div className="space-y-4 border-t pt-4">
            <h3 className="text-sm font-medium">Notification Types</h3>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="text-sm font-normal">Meetup Nudges</Label>
                  <p className="text-xs text-muted-foreground">
                    Get notified when compatible dogs are nearby
                  </p>
                </div>
                <Switch defaultChecked disabled={isLoading} />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="text-sm font-normal">Event Reminders</Label>
                  <p className="text-xs text-muted-foreground">
                    Reminders for upcoming dog events you're attending
                  </p>
                </div>
                <Switch defaultChecked disabled={isLoading} />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="text-sm font-normal">Achievements</Label>
                  <p className="text-xs text-muted-foreground">
                    Celebrate milestones and trophies earned
                  </p>
                </div>
                <Switch defaultChecked disabled={isLoading} />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="text-sm font-normal">New Messages</Label>
                  <p className="text-xs text-muted-foreground">
                    Get notified about new chat messages
                  </p>
                </div>
                <Switch defaultChecked disabled={isLoading} />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="text-sm font-normal">Service Recommendations</Label>
                  <p className="text-xs text-muted-foreground">
                    Discover nearby pet services and recommendations
                  </p>
                </div>
                <Switch defaultChecked disabled={isLoading} />
              </div>
            </div>
          </div>
        )}

        {/* Test Notification Button (Development) */}
        {isSubscribed && process.env.NODE_ENV === "development" && (
          <div className="border-t pt-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                new Notification("Test Notification", {
                  body: "This is a test notification from Woof!",
                  icon: "/icon-192.png",
                  badge: "/icon-192.png",
                })
              }}
            >
              Send Test Notification
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
