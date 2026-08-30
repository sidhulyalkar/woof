export const AUTH_STORAGE_KEY = 'woof-auth-storage';
export const LEGACY_SESSION_STORAGE_KEY = 'woof-session-storage';
export const AUTH_PERSIST_VERSION = 0 as const;

export type PersistedAuthSnapshot<TUser> = {
  state: {
    user: TUser;
    token: string;
    isAuthenticated: true;
  };
  version: typeof AUTH_PERSIST_VERSION;
};

/**
 * Serializes the exact persisted Zustand payload owned by the production auth store.
 * Browser tests consume this helper rather than hand-maintaining a second schema.
 */
export function serializePersistedAuthSession<TUser>(user: TUser, token: string): string {
  const payload: PersistedAuthSnapshot<TUser> = {
    state: {
      user,
      token,
      isAuthenticated: true,
    },
    version: AUTH_PERSIST_VERSION,
  };

  return JSON.stringify(payload);
}
