'use client';

import { BellOff } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export function NotificationSettings() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BellOff className="h-5 w-5" aria-hidden="true" />
          Browser notifications
        </CardTitle>
        <CardDescription>Not enabled in this Web release</CardDescription>
      </CardHeader>
      <CardContent>
        <Alert>
          <AlertDescription>
            Woof does not install an application service worker in the current browser release, so
            background Web Push is intentionally unavailable. Updates remain available inside Woof;
            native notification delivery is qualified separately from the Web client.
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}
