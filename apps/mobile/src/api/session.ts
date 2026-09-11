import * as SecureStore from 'expo-secure-store';

export const ACCESS_TOKEN_KEY = 'woofAccessToken';

export type SessionInvalidation = {
  reason: 'unauthorized';
  rejectedToken: string;
};

type SessionInvalidationListener = (event: SessionInvalidation) => void;

type ConditionalClearResult = {
  matched: boolean;
  cleared: boolean;
};

const invalidationListeners = new Set<SessionInvalidationListener>();
let credentialTail: Promise<void> = Promise.resolve();

function withCredentialLock<T>(operation: () => Promise<T>): Promise<T> {
  const run = credentialTail.then(operation, operation);
  credentialTail = run.then(
    () => undefined,
    () => undefined
  );
  return run;
}

export function getAccessToken(): Promise<string | null> {
  return withCredentialLock(() => SecureStore.getItemAsync(ACCESS_TOKEN_KEY));
}

export function storeAccessToken(token: string): Promise<void> {
  return withCredentialLock(() => SecureStore.setItemAsync(ACCESS_TOKEN_KEY, token));
}

export function takeAccessToken(): Promise<string | null> {
  return withCredentialLock(async () => {
    const token = await SecureStore.getItemAsync(ACCESS_TOKEN_KEY);
    if (token) {
      await SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY);
    }
    return token;
  });
}

export function clearAccessToken(): Promise<void> {
  return withCredentialLock(() => SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY));
}

export function clearAccessTokenIfCurrent(expectedToken: string): Promise<ConditionalClearResult> {
  return withCredentialLock(async () => {
    const currentToken = await SecureStore.getItemAsync(ACCESS_TOKEN_KEY);
    if (currentToken !== expectedToken) {
      return { matched: false, cleared: false };
    }

    try {
      await SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY);
      return { matched: true, cleared: true };
    } catch {
      return { matched: true, cleared: false };
    }
  });
}

export function rejectAccessTokenIfCurrent(rejectedToken: string): Promise<ConditionalClearResult> {
  return clearAccessTokenIfCurrent(rejectedToken);
}

export function subscribeToSessionInvalidation(listener: SessionInvalidationListener): () => void {
  invalidationListeners.add(listener);
  return () => invalidationListeners.delete(listener);
}

export function publishSessionInvalidation(event: SessionInvalidation): void {
  for (const listener of invalidationListeners) {
    listener(event);
  }
}
