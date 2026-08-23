import { ConciergeService } from './concierge.service';

describe('ConciergeService', () => {
  const adventure = { getDashboard: jest.fn() };
  const careEvents = { getSummary: jest.fn() };
  const autopilot = { getDashboard: jest.fn() };
  const connectors = { getDashboard: jest.fn() };
  const service = new ConciergeService(
    adventure as never,
    careEvents as never,
    autopilot as never,
    connectors as never
  );

  const baseAdventure = {
    pet: { id: 'pet-1', name: 'Scout', species: 'DOG' },
    quests: [
      {
        id: 'quest-1',
        title: 'Take a Sniffari',
        why: 'Exploration adds variety without making distance the objective.',
        primaryPathway: 'EXPLORE',
        actionLabel: 'Start an exploration',
        href: '/activity',
      },
    ],
  };

  beforeEach(() => {
    jest.clearAllMocks();
    adventure.getDashboard.mockResolvedValue(baseAdventure);
    careEvents.getSummary.mockResolvedValue({ recentEvents: [] });
    autopilot.getDashboard.mockResolvedValue({ reminders: [], signals: [] });
    connectors.getDashboard.mockResolvedValue({
      providers: [
        { provider: 'FI', label: 'Fi', availability: 'PARTNER_REQUIRED' },
        { provider: 'TRACTIVE', label: 'Tractive', availability: 'PARTNER_REQUIRED' },
      ],
    });
  });

  it('fails closed on live weather and keeps the briefing suggestion-only', async () => {
    const result = await service.getToday('user-1');

    expect(result.context.weather).toEqual(
      expect.objectContaining({ status: 'NOT_CONFIGURED', live: false })
    );
    expect(result.boundaries).toEqual(
      expect.objectContaining({
        suggestionOnly: true,
        liveWeatherUsed: false,
        diagnosticInferenceAllowed: false,
        prescriptionOrDoseCalculationAllowed: false,
        persistentStateMutationAllowed: false,
        autonomousPurchaseAllowed: false,
      })
    );
    expect(result.briefing.topQuest).toEqual(
      expect.objectContaining({
        title: 'Take a Sniffari',
        reason: expect.any(String),
        evidence: expect.arrayContaining([expect.objectContaining({ source: 'ADVENTURE' })]),
      })
    );
  });

  it('uses recent explicit owner feedback to lower pace without claiming a mood inference', async () => {
    careEvents.getSummary.mockResolvedValue({
      recentEvents: [
        {
          id: 'care-1',
          occurredAt: new Date().toISOString(),
          outcome: { ownerExperience: 'a_lot_today', dogExperience: 'comfortable' },
        },
      ],
    });

    const result = await service.getToday('user-1');

    expect(result.context.pace.mode).toBe('GENTLE');
    expect(result.context.pace.reason).toContain('explicit feedback');
    expect(result.context.pace.reason.toLowerCase()).not.toContain('diagnos');
    expect(result.suggestions).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: 'RECOVERY_PACE',
          suggestionOnly: true,
          evidence: expect.arrayContaining([
            expect.objectContaining({ source: 'CARE_EVENT', referenceId: 'care-1' }),
          ]),
        }),
      ])
    );
  });

  it('does not let old negative feedback permanently suppress the current pace', async () => {
    careEvents.getSummary.mockResolvedValue({
      recentEvents: [
        {
          id: 'care-old',
          occurredAt: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString(),
          outcome: { dogExperience: 'not_their_thing', ownerExperience: 'fine' },
        },
      ],
    });

    const result = await service.getToday('user-1');

    expect(result.context.pace.mode).toBe('NORMAL');
    expect(result.suggestions.find((item) => item.kind === 'RECOVERY_PACE')).toBeUndefined();
  });

  it('surfaces medication prep without calculating dosage or inventing instructions', async () => {
    autopilot.getDashboard.mockResolvedValue({
      reminders: [
        {
          id: 'reminder-1',
          kind: 'MEDICATION',
          title: 'Monthly medication',
          petId: 'pet-1',
          dueAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
          status: 'SCHEDULED',
        },
      ],
      signals: [],
    });

    const result = await service.getToday('user-1');
    const suggestion = result.suggestions.find((item) => item.kind === 'CARE_PREP');

    expect(suggestion).toEqual(
      expect.objectContaining({
        priority: 'ATTENTION',
        reason: expect.any(String),
        evidence: expect.arrayContaining([
          expect.objectContaining({ source: 'AUTOPILOT', referenceId: 'reminder-1' }),
        ]),
      })
    );
    expect(suggestion?.body).toContain('does not calculate doses');
    expect(suggestion?.body).not.toMatch(/\b\d+(\.\d+)?\s?(mg|ml|mcg|tablet|capsule)s?\b/i);
  });

  it('passes through tracker check-ins as explicitly non-diagnostic context', async () => {
    autopilot.getDashboard.mockResolvedValue({
      reminders: [],
      signals: [
        {
          id: 'signal-1',
          petId: 'pet-1',
          title: 'A quieter activity day',
          body: 'Today is lower than Scout’s recent tracker pattern. Check in on how the day feels.',
          observedAt: new Date().toISOString(),
          nonDiagnostic: true,
        },
      ],
    });

    const result = await service.getToday('user-1');
    const suggestion = result.suggestions.find((item) => item.kind === 'CHECK_IN');

    expect(suggestion?.reason).toContain('non-diagnostic');
    expect(suggestion?.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ source: 'AUTOPILOT', referenceId: 'signal-1' }),
      ])
    );
  });

  it('surfaces stale connector authentication as context rather than stale provider truth', async () => {
    connectors.getDashboard.mockResolvedValue({
      providers: [{ provider: 'FI', label: 'Fi', availability: 'REAUTH_REQUIRED' }],
    });

    const result = await service.getToday('user-1');
    const suggestion = result.suggestions.find((item) => item.kind === 'CONNECTION_ATTENTION');

    expect(suggestion).toEqual(
      expect.objectContaining({
        priority: 'INFO',
        action: { label: 'View connected services', href: '/connectors' },
        suggestionOnly: true,
      })
    );
    expect(suggestion?.body).toContain('will not pretend stale provider access is current');
  });

  it('requires reason and evidence on every surfaced suggestion', async () => {
    careEvents.getSummary.mockResolvedValue({
      recentEvents: [
        {
          id: 'care-2',
          occurredAt: new Date().toISOString(),
          outcome: { safeOptOut: true },
        },
      ],
    });
    autopilot.getDashboard.mockResolvedValue({
      reminders: [],
      signals: [
        {
          id: 'signal-2',
          petId: 'pet-1',
          title: 'Tracker battery is low',
          body: 'Charge the tracker when convenient.',
          observedAt: new Date().toISOString(),
          nonDiagnostic: true,
        },
      ],
    });
    connectors.getDashboard.mockResolvedValue({
      providers: [{ provider: 'FI', label: 'Fi', availability: 'REAUTH_REQUIRED' }],
    });

    const result = await service.getToday('user-1');

    expect(result.suggestions.length).toBeGreaterThan(0);
    for (const suggestion of result.suggestions) {
      expect(suggestion.reason.length).toBeGreaterThan(10);
      expect(suggestion.evidence.length).toBeGreaterThan(0);
      expect(suggestion.suggestionOnly).toBe(true);
    }
  });
});
