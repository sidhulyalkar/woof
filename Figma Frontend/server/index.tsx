import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import * as kv from "./kv_store.tsx";

const app = new Hono();

// Enable logger
app.use('*', logger(console.log));

// Enable CORS for all routes and methods
app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

// Error handler
app.onError((err, c) => {
  console.error('Server error:', err);
  return c.json({ error: 'Internal server error', message: err.message }, 500);
});

// Health check endpoint
app.get("/make-server-ec56cf0b/health", async (c) => {
  try {
    // Test KV store connection
    await kv.get("health_check");
    return c.json({ 
      status: "ok", 
      timestamp: new Date().toISOString(),
      kv_connection: "healthy"
    });
  } catch (error) {
    console.error('Health check failed:', error);
    return c.json({ 
      status: "error", 
      error: error.message,
      timestamp: new Date().toISOString() 
    }, 500);
  }
});

// Test route for KV store
app.get("/make-server-ec56cf0b/test", async (c) => {
  try {
    const testKey = "test_connection";
    const testValue = { timestamp: new Date().toISOString(), message: "test successful" };
    
    await kv.set(testKey, testValue);
    const retrieved = await kv.get(testKey);
    
    return c.json({ 
      success: true, 
      data: retrieved,
      message: "KV store working correctly"
    });
  } catch (error) {
    console.error('Test route error:', error);
    return c.json({ 
      success: false, 
      error: error.message 
    }, 500);
  }
});

// Feed endpoints
app.get("/make-server-ec56cf0b/feed", async (c) => {
  try {
    const posts = await kv.getByPrefix("post_");
    return c.json({ posts: posts || [] });
  } catch (error) {
    console.error('Feed error:', error);
    return c.json({ error: error.message }, 500);
  }
});

app.post("/make-server-ec56cf0b/feed", async (c) => {
  try {
    const body = await c.req.json();
    const postId = `post_${Date.now()}`;
    await kv.set(postId, { ...body, id: postId, createdAt: new Date().toISOString() });
    return c.json({ success: true, id: postId });
  } catch (error) {
    console.error('Post creation error:', error);
    return c.json({ error: error.message }, 500);
  }
});

// Activity endpoints
app.get("/make-server-ec56cf0b/activities", async (c) => {
  try {
    const activities = await kv.getByPrefix("activity_");
    return c.json({ activities: activities || [] });
  } catch (error) {
    console.error('Activities error:', error);
    return c.json({ error: error.message }, 500);
  }
});

app.post("/make-server-ec56cf0b/activities", async (c) => {
  try {
    const body = await c.req.json();
    const activityId = `activity_${Date.now()}`;
    await kv.set(activityId, { ...body, id: activityId, createdAt: new Date().toISOString() });
    return c.json({ success: true, id: activityId });
  } catch (error) {
    console.error('Activity creation error:', error);
    return c.json({ error: error.message }, 500);
  }
});

// Leaderboard endpoints
app.get("/make-server-ec56cf0b/leaderboard", async (c) => {
  try {
    const timeframe = c.req.query('timeframe') || 'weekly';
    const leaderboard = await kv.get(`leaderboard_${timeframe}`);
    return c.json({ leaderboard: leaderboard || [] });
  } catch (error) {
    console.error('Leaderboard error:', error);
    return c.json({ error: error.message }, 500);
  }
});

// Messages endpoints
app.get("/make-server-ec56cf0b/messages", async (c) => {
  try {
    const messages = await kv.getByPrefix("message_");
    return c.json({ messages: messages || [] });
  } catch (error) {
    console.error('Messages error:', error);
    return c.json({ error: error.message }, 500);
  }
});

app.post("/make-server-ec56cf0b/messages", async (c) => {
  try {
    const body = await c.req.json();
    const messageId = `message_${Date.now()}`;
    await kv.set(messageId, { ...body, id: messageId, createdAt: new Date().toISOString() });
    return c.json({ success: true, id: messageId });
  } catch (error) {
    console.error('Message creation error:', error);
    return c.json({ error: error.message }, 500);
  }
});

// Profile endpoints
app.get("/make-server-ec56cf0b/profile/:userId", async (c) => {
  try {
    const userId = c.req.param('userId');
    const profile = await kv.get(`profile_${userId}`);
    return c.json({ profile: profile || null });
  } catch (error) {
    console.error('Profile error:', error);
    return c.json({ error: error.message }, 500);
  }
});

app.put("/make-server-ec56cf0b/profile/:userId", async (c) => {
  try {
    const userId = c.req.param('userId');
    const body = await c.req.json();
    await kv.set(`profile_${userId}`, { ...body, userId, updatedAt: new Date().toISOString() });
    return c.json({ success: true });
  } catch (error) {
    console.error('Profile update error:', error);
    return c.json({ error: error.message }, 500);
  }
});

// 404 handler
app.notFound((c) => {
  return c.json({ error: 'Route not found' }, 404);
});

console.log('PetPath server starting...');
Deno.serve(app.fetch);