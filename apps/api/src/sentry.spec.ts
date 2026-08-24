import { scrubSentryEvent, scrubSentryTransaction } from './sentry';

describe('Sentry privacy scrubbing', () => {
  it('removes request, user, extra and breadcrumb payloads from error events', () => {
    const event = {
      request: {
        url: '/api/v1/users/user-secret?token=secret-token',
        headers: { authorization: 'Bearer secret-token' },
      },
      user: { id: 'user-secret', email: 'private@example.com' },
      extra: { requestBody: { email: 'private@example.com' } },
      breadcrumbs: [{ data: { url: '/private' } }],
      contexts: { http: { route: '/api/v1/users/:userId' } },
    };

    expect(scrubSentryEvent(event)).toBe(event);
    expect(event.request).toBeUndefined();
    expect(event.user).toBeUndefined();
    expect(event.extra).toBeUndefined();
    expect(event.breadcrumbs).toBeUndefined();
    expect(event.contexts).toEqual({ http: { route: '/api/v1/users/:userId' } });
  });

  it('keeps span timing shape but removes span data and high-cardinality descriptions', () => {
    const event = {
      request: { url: '/secret' },
      spans: [
        {
          op: 'db.sql.prisma',
          description: 'SELECT * FROM users WHERE email = private@example.com',
          data: { 'db.query': 'SELECT * FROM users WHERE email = private@example.com' },
        },
        {
          op: 'http.client',
          description: 'https://provider.example/member/user-secret?token=secret-token',
          data: { url: 'https://provider.example/member/user-secret?token=secret-token' },
        },
      ],
    };

    scrubSentryTransaction(event);

    expect(event.request).toBeUndefined();
    expect(event.spans).toEqual([
      { op: 'db.sql.prisma', description: 'db.sql.prisma' },
      { op: 'http.client', description: 'http.client' },
    ]);
    expect(JSON.stringify(event)).not.toContain('private@example.com');
    expect(JSON.stringify(event)).not.toContain('secret-token');
    expect(JSON.stringify(event)).not.toContain('user-secret');
  });
});
