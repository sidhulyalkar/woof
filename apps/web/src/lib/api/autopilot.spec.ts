import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  delete: vi.fn(),
}));

vi.mock('./client', () => ({
  apiClient: transport,
}));

import { autopilotApi, type CreateCareReminderInput } from './autopilot';

describe('autopilotApi', () => {
  beforeEach(() => {
    transport.get.mockReset();
    transport.post.mockReset();
    transport.delete.mockReset();
  });

  it('reads the authenticated Autopilot dashboard', async () => {
    transport.get.mockResolvedValue({ reminders: [], signals: [] });

    await autopilotApi.getDashboard();

    expect(transport.get).toHaveBeenCalledWith('/autopilot');
  });

  it('preserves pet-scoped reminder input without adding autonomous care fields', async () => {
    transport.post.mockResolvedValue({ id: 'reminder-1' });
    const input: CreateCareReminderInput = {
      petId: 'pet-1',
      kind: 'MEDICATION',
      title: 'Heartworm medication',
      detail: 'Use the veterinarian-provided instructions.',
      dueAt: '2026-08-24T18:00:00.000Z',
      repeatEveryDays: 30,
    };

    await autopilotApi.createReminder(input);

    expect(transport.post).toHaveBeenCalledWith('/autopilot/reminders', input);
    const body = transport.post.mock.calls[0]?.[1] as Record<string, unknown>;
    expect(body).not.toHaveProperty('dose');
    expect(body).not.toHaveProperty('dosage');
    expect(body).not.toHaveProperty('administerAutomatically');
  });

  it('cancels only the selected reminder route', async () => {
    transport.delete.mockResolvedValue({ success: true });

    await autopilotApi.cancelReminder('reminder-1');

    expect(transport.delete).toHaveBeenCalledWith('/autopilot/reminders/reminder-1');
  });

  it('acknowledges signals without mutating the underlying observation', async () => {
    transport.post.mockResolvedValue({ success: true });

    await autopilotApi.acknowledgeSignal('signal-1');

    expect(transport.post).toHaveBeenCalledWith('/autopilot/signals/signal-1/acknowledge');
  });
});
