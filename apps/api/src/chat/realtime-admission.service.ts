import { Injectable } from '@nestjs/common';

export type RealtimeAction = 'message' | 'typing' | 'membership';

export type RealtimeAdmissionDecision =
  { allowed: true } | { allowed: false; retryAfterMs: number };

type WindowPolicy = {
  limit: number;
  windowMs: number;
};

type Bucket = {
  timestamps: number[];
  lastSeenAt: number;
};

export const REALTIME_ADMISSION_POLICIES: Record<RealtimeAction, WindowPolicy[]> = {
  message: [
    { limit: 5, windowMs: 1_000 },
    { limit: 60, windowMs: 60_000 },
  ],
  typing: [
    { limit: 8, windowMs: 5_000 },
    { limit: 60, windowMs: 60_000 },
  ],
  membership: [
    { limit: 10, windowMs: 5_000 },
    { limit: 60, windowMs: 60_000 },
  ],
};

const MAX_REALTIME_BUCKETS = 10_000;
const PRUNE_INTERVAL = 256;
const MAX_WINDOW_MS = Math.max(
  ...Object.values(REALTIME_ADMISSION_POLICIES).flatMap((policies) =>
    policies.map((policy) => policy.windowMs)
  )
);

@Injectable()
export class RealtimeAdmissionService {
  private readonly buckets = new Map<string, Bucket>();
  private operations = 0;

  consume(userId: string, action: RealtimeAction, now = Date.now()): RealtimeAdmissionDecision {
    this.operations += 1;
    if (this.operations % PRUNE_INTERVAL === 0) {
      this.pruneExpired(now);
    }

    const key = `${action}:${userId}`;
    let bucket = this.buckets.get(key);
    if (!bucket) {
      this.ensureCapacity(now);
      bucket = { timestamps: [], lastSeenAt: now };
      this.buckets.set(key, bucket);
    }

    bucket.timestamps = bucket.timestamps.filter((timestamp) => timestamp > now - MAX_WINDOW_MS);
    bucket.lastSeenAt = now;

    for (const policy of REALTIME_ADMISSION_POLICIES[action]) {
      const windowStart = now - policy.windowMs;
      const recent = bucket.timestamps.filter((timestamp) => timestamp > windowStart);
      if (recent.length >= policy.limit) {
        return {
          allowed: false,
          retryAfterMs: Math.max(1, recent[0]! + policy.windowMs - now),
        };
      }
    }

    bucket.timestamps.push(now);
    return { allowed: true };
  }

  private ensureCapacity(now: number) {
    if (this.buckets.size < MAX_REALTIME_BUCKETS) return;
    this.pruneExpired(now);

    while (this.buckets.size >= MAX_REALTIME_BUCKETS) {
      let oldestKey: string | null = null;
      let oldestSeenAt = Number.POSITIVE_INFINITY;

      for (const [key, bucket] of this.buckets) {
        if (bucket.lastSeenAt < oldestSeenAt) {
          oldestSeenAt = bucket.lastSeenAt;
          oldestKey = key;
        }
      }

      if (!oldestKey) return;
      this.buckets.delete(oldestKey);
    }
  }

  private pruneExpired(now: number) {
    for (const [key, bucket] of this.buckets) {
      bucket.timestamps = bucket.timestamps.filter((timestamp) => timestamp > now - MAX_WINDOW_MS);
      if (bucket.timestamps.length === 0) {
        this.buckets.delete(key);
      }
    }
  }
}
