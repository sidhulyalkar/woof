// Service Worker for PetPath PWA
const CACHE_NAME = "petpath-v1"
const urlsToCache = [
  "/",
  "/matches",
  "/messages",
  "/events",
  "/profile",
  "/friends",
  "/health",
  "/wellness",
  "/map",
  "/settings",
]

// Install event - cache essential resources
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("[v0] Service Worker: Caching app shell")
      return cache.addAll(urlsToCache)
    }),
  )
  self.skipWaiting()
})

// Activate event - clean up old caches
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log("[v0] Service Worker: Removing old cache", cacheName)
            return caches.delete(cacheName)
          }
        }),
      )
    }),
  )
  self.clients.claim()
})

// Fetch event - serve from cache, fallback to network
self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      // Cache hit - return response
      if (response) {
        return response
      }

      // Clone the request
      const fetchRequest = event.request.clone()

      return fetch(fetchRequest).then((response) => {
        // Check if valid response
        if (!response || response.status !== 200 || response.type !== "basic") {
          return response
        }

        // Clone the response
        const responseToCache = response.clone()

        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache)
        })

        return response
      })
    }),
  )
})

// Push notification event - display notification
self.addEventListener("push", (event) => {
  console.log("[v0] Service Worker: Push notification received", event)

  let data = {
    title: "Woof 🐾",
    body: "You have a new notification",
    icon: "/icon-192.png",
    badge: "/icon-192.png",
    data: { url: "/" },
  }

  if (event.data) {
    try {
      data = event.data.json()
    } catch (e) {
      console.error("[v0] Service Worker: Error parsing push data", e)
    }
  }

  const options = {
    body: data.body,
    icon: data.icon || "/icon-192.png",
    badge: data.badge || "/icon-192.png",
    tag: data.tag || "default",
    data: data.data || { url: "/" },
    vibrate: [200, 100, 200],
    requireInteraction: false,
  }

  event.waitUntil(self.registration.showNotification(data.title, options))
})

// Notification click event - open app
self.addEventListener("notificationclick", (event) => {
  console.log("[v0] Service Worker: Notification clicked", event)

  event.notification.close()

  const urlToOpen = event.notification.data?.url || "/"

  event.waitUntil(
    clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then((clientList) => {
        // Check if app is already open
        for (const client of clientList) {
          if (client.url.includes(urlToOpen) && "focus" in client) {
            return client.focus()
          }
        }
        // Open new window
        if (clients.openWindow) {
          return clients.openWindow(urlToOpen)
        }
      }),
  )
})
