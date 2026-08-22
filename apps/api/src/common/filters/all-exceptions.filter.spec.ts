import { redactRequestHeaders } from './all-exceptions.filter';

describe('redactRequestHeaders', () => {
  it('removes credentials while preserving non-sensitive diagnostic headers', () => {
    expect(
      redactRequestHeaders({
        authorization: 'Bearer top-secret',
        cookie: 'session=secret',
        'x-api-key': 'private-key',
        'user-agent': 'woof-test-agent',
        'x-request-id': 'request-123',
      })
    ).toEqual({
      'user-agent': 'woof-test-agent',
      'x-request-id': 'request-123',
    });
  });
});
