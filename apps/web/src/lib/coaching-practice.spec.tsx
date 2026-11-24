import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, expect, it, vi } from 'vitest';
import CoachPage from '../app/coach/page';
import { coachingApi, type CoachingDashboard } from './api/coaching';
vi.mock('./api/coaching', () => ({ coachingApi: { getMine: vi.fn(), recordSession: vi.fn() } }));
vi.mock('../components/bottom-nav', () => ({ BottomNav: () => null }));
const decision = {
  action: 'hold' as const,
  nextLevel: 1,
  headline: 'Repeat',
  reason: 'One step at a time',
};
const support = { recommended: false, message: 'Keep it comfortable' };
const plan = {
  id: 'plan',
  petId: 'dog',
  status: 'ACTIVE',
  title: 'Check in',
  skill: 'attention',
  objective: 'Notice attention',
  cue: 'Name',
  handlerFocus: 'Mark a glance',
  steps: [],
  rewardExamples: [],
  safety: '',
  level: 1,
  levelLabel: 'Quiet',
  targetSuccessRate: 0.8,
  sessionsCompleted: 0,
  practiceCoverage: 0,
  recentSuccessRate: null,
  lastPracticedAt: null,
  nextPractice: decision,
  support,
  updatedAt: '2026-11-22T20:00:00.000Z',
};
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(coachingApi.getMine).mockResolvedValue({
    pet: { id: 'dog', name: 'Shasta', species: 'DOG' },
    activePlan: plan,
    pausedPlans: [],
    templates: [],
    weeklyRhythm: { sessions: 0, minutes: 0 },
    methodology: { version: 'v1', principles: [], progressionPolicy: '', sources: [] },
  } satisfies CoachingDashboard);
});
async function openPractice() {
  render(
    <QueryClientProvider
      client={
        new QueryClient({
          defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
        })
      }
    >
      <CoachPage />
    </QueryClientProvider>
  );
  fireEvent.click(await screen.findByText('Start a short practice'));
}
it('freezes the full request across a lost-response retry and does not invent distraction', async () => {
  vi.mocked(coachingApi.recordSession)
    .mockRejectedValueOnce(new Error('lost'))
    .mockResolvedValueOnce({ activityId: 'activity', plan, decision, support });
  await openPractice();
  fireEvent.click(screen.getByText('Nice!'));
  fireEvent.click(screen.getByText('Save practice'));
  fireEvent.click(await screen.findByText('Retry the same save'));
  await waitFor(() => expect(coachingApi.recordSession).toHaveBeenCalledTimes(2));
  const calls = vi.mocked(coachingApi.recordSession).mock.calls;
  expect(calls[1]).toEqual(calls[0]);
  expect(calls[0]![1]).toMatchObject({ expectedPlanUpdatedAt: plan.updatedAt, attempts: 1 });
  expect(calls[0]![1].sessionKey).toMatch(/^[a-f0-9-]{36}$/);
  expect(calls[0]![1].distractionLevel).toBeUndefined();
});
it('allows stopping before any repetition and records the chosen environment', async () => {
  vi.mocked(coachingApi.recordSession).mockResolvedValue({
    activityId: 'activity',
    plan,
    decision,
    support,
  });
  await openPractice();
  fireEvent.change(screen.getByLabelText('How distracting was the setting?'), {
    target: { value: '2' },
  });
  fireEvent.click(screen.getByText('We stopped early'));
  fireEvent.click(screen.getByText('Save practice'));
  await waitFor(() =>
    expect(coachingApi.recordSession).toHaveBeenCalledWith(
      'plan',
      expect.objectContaining({
        attempts: 0,
        successes: 0,
        stoppedEarly: true,
        distractionLevel: 2,
      })
    )
  );
});
