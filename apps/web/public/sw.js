// Transitional cleanup worker for legacy Woof browser caches.
//
// Woof no longer registers a service worker in the current release. Existing
// browsers may still have the old cache-first worker installed, so this file
// intentionally updates that registration once, removes legacy caches, and
// unregisters itself. It must not cache or serve application/user data.

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    Promise.all([
      caches
        .keys()
        .then((cacheNames) =>
          Promise.all(
            cacheNames
              .filter((cacheName) => cacheName.startsWith('petpath-'))
              .map((cacheName) => caches.delete(cacheName))
          )
        ),
      self.registration.unregister(),
      self.clients.claim(),
    ])
  );
});
