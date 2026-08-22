declare global {
  interface Window {
    google?: {
      accounts?: {
        oauth2?: {
          initTokenClient: (config: {
            client_id: string;
            scope: string;
            callback: (response: { access_token?: string; error?: string }) => void;
            error_callback?: (error: unknown) => void;
          }) => { requestAccessToken: (options?: { prompt?: string }) => void };
        };
      };
    };
  }
}

const GOOGLE_SCRIPT_ID = 'woof-google-identity-services';

async function loadGoogleIdentityServices() {
  if (window.google?.accounts?.oauth2) return;
  await new Promise<void>((resolve, reject) => {
    const existing = document.getElementById(GOOGLE_SCRIPT_ID) as HTMLScriptElement | null;
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error('Google sign-in script failed to load')), {
        once: true,
      });
      return;
    }

    const script = document.createElement('script');
    script.id = GOOGLE_SCRIPT_ID;
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Google sign-in script failed to load'));
    document.head.appendChild(script);
  });
}

export async function requestGooglePhotosToken(mode: 'import' | 'export') {
  const clientId = process.env.NEXT_PUBLIC_GOOGLE_PHOTOS_CLIENT_ID;
  if (!clientId) {
    throw new Error('Google Photos OAuth is not configured for this deployment.');
  }
  await loadGoogleIdentityServices();
  const oauth = window.google?.accounts?.oauth2;
  if (!oauth) throw new Error('Google OAuth is unavailable in this browser.');

  const scope =
    mode === 'import'
      ? 'https://www.googleapis.com/auth/photospicker.mediaitems.readonly'
      : 'https://www.googleapis.com/auth/photoslibrary.appendonly';

  return new Promise<string>((resolve, reject) => {
    const client = oauth.initTokenClient({
      client_id: clientId,
      scope,
      callback: (response) => {
        if (response.access_token) resolve(response.access_token);
        else reject(new Error(response.error || 'Google Photos authorization was not completed.'));
      },
      error_callback: () => reject(new Error('Google Photos authorization was cancelled.')),
    });
    client.requestAccessToken({ prompt: '' });
  });
}

export {};
