import { Logger, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
  HealthAiService,
  normalizeHealthModelResult,
  type PetHealthModelResult,
} from './health-ai.service';

function assessment(overrides: Partial<PetHealthModelResult> = {}): PetHealthModelResult {
  return {
    triage: 'monitor',
    confidence: 0.72,
    summary: 'A localized visible change is present, but the cause cannot be determined here.',
    visibleFindings: ['small area of redness'],
    possibleCategories: ['dermatologic irritation'],
    photoFeedback: {
      usable: true,
      reason: 'The area is visible.',
      betterPhotoInstructions: [],
    },
    questions: ['Is it changing quickly?'],
    ownerActions: ['Document whether the area changes over the next several hours.'],
    avoid: ['Do not use human medication based on an automated screening result.'],
    vetHandoff: {
      recommended: false,
      timing: 'not-yet',
      summary: 'Monitor and contact your veterinarian if the concern persists or worsens.',
      bring: [],
    },
    ...overrides,
  };
}

function service(overrides: Record<string, string | undefined> = {}) {
  const values: Record<string, string | undefined> = {
    OPENAI_API_KEY: 'openai-test-key-that-is-long-enough',
    OPENAI_HEALTH_MODEL: 'health-test-model',
    OPENAI_HEALTH_TIMEOUT_MS: '12000',
    ...overrides,
  };
  const config = {
    get: jest.fn((key: string) => values[key]),
  };
  return new HealthAiService(config as unknown as ConfigService);
}

const input = {
  pet: {
    name: 'Nova',
    species: 'DOG',
    breed: null,
    ageYears: 4,
    temperament: null,
  },
  concern: 'A small red patch appeared today.',
  recentContext: [],
  priorHealthObservations: [],
};

function providerResponse(result: PetHealthModelResult = assessment()) {
  return new Response(
    JSON.stringify({
      output: [
        {
          content: [{ type: 'output_text', text: JSON.stringify(result) }],
        },
      ],
    }),
    { status: 200 }
  );
}

function loggerSpies() {
  return {
    warn: jest.spyOn(Logger.prototype, 'warn').mockImplementation(() => undefined),
    error: jest.spyOn(Logger.prototype, 'error').mockImplementation(() => undefined),
  };
}

function serializedLogCalls(spies: ReturnType<typeof loggerSpies>) {
  return JSON.stringify([...spies.warn.mock.calls, ...spies.error.mock.calls]);
}

describe('Health Lens model output authority', () => {
  it('raises an emergency model result to an authoritative immediate handoff', () => {
    const normalized = normalizeHealthModelResult(
      assessment({
        triage: 'emergency_now',
        vetHandoff: {
          recommended: false,
          timing: 'not-yet',
          summary: 'Model handoff was internally inconsistent.',
          bring: [],
        },
      })
    );

    expect(normalized.vetHandoff.recommended).toBe(true);
    expect(normalized.vetHandoff.timing).toBe('now');
    expect(normalized.ownerActions[0]).toMatch(/emergency veterinarian now/i);
  });

  it('does not let vet-today or vet-soon triage carry a lower-urgency handoff', () => {
    const today = normalizeHealthModelResult(
      assessment({
        triage: 'vet_today',
        vetHandoff: {
          recommended: false,
          timing: 'routine',
          summary: 'Needs assessment.',
          bring: [],
        },
      })
    );
    const soon = normalizeHealthModelResult(
      assessment({
        triage: 'vet_soon',
        vetHandoff: {
          recommended: false,
          timing: 'not-yet',
          summary: 'Needs assessment.',
          bring: [],
        },
      })
    );

    expect(today.vetHandoff).toMatchObject({ recommended: true, timing: 'today' });
    expect(soon.vetHandoff).toMatchObject({
      recommended: true,
      timing: 'within-2-days',
    });
  });

  it('makes better-image triage authoritative over a contradictory usable-photo flag', () => {
    const normalized = normalizeHealthModelResult(
      assessment({
        triage: 'better_image',
        photoFeedback: {
          usable: true,
          reason: 'The model nevertheless requested a better image.',
          betterPhotoInstructions: ['Use even lighting and include the surrounding area.'],
        },
      })
    );

    expect(normalized.photoFeedback.usable).toBe(false);
  });

  it('fails closed on positive medication, dosing, prescription, vomiting, or invasive directives', () => {
    const unsafe = [
      'Give 25 mg of a painkiller now.',
      'Administer the medication twice today.',
      'Stop the prescription until the swelling improves.',
      'Induce vomiting at home.',
      'Drain the lesion and keep it clean.',
    ];

    for (const directive of unsafe) {
      expect(() => normalizeHealthModelResult(assessment({ ownerActions: [directive] }))).toThrow(
        ServiceUnavailableException
      );
    }
  });

  it('fails closed when unsafe treatment instructions migrate into other visible fields', () => {
    const unsafeCases: Array<Partial<PetHealthModelResult>> = [
      { summary: 'Give 25 mg of aspirin now.' },
      { questions: ['Could you administer the medication now?'] },
      {
        photoFeedback: {
          usable: false,
          reason: 'Apply ibuprofen before taking the next photo.',
          betterPhotoInstructions: [],
        },
      },
      {
        vetHandoff: {
          recommended: true,
          timing: 'today',
          summary: 'Give aspirin while waiting for the appointment.',
          bring: [],
        },
      },
      {
        vetHandoff: {
          recommended: true,
          timing: 'today',
          summary: 'Arrange veterinary assessment today.',
          bring: ['Administer the medication before travel.'],
        },
      },
    ];

    for (const overrides of unsafeCases) {
      expect(() => normalizeHealthModelResult(assessment(overrides))).toThrow(
        ServiceUnavailableException
      );
    }
  });

  it('allows explicit avoid/negated safety language while bounding generated arrays', () => {
    const normalized = normalizeHealthModelResult(
      assessment({
        ownerActions: ['Do not give human medication without veterinary guidance.'],
        visibleFindings: Array.from({ length: 20 }, (_, index) => `finding-${index}`),
      })
    );

    expect(normalized.ownerActions).toEqual([
      'Do not give human medication without veterinary guidance.',
    ]);
    expect(normalized.visibleFindings).toHaveLength(8);
  });

  it('rejects malformed nested contracts even when top-level triage is valid', () => {
    expect(() =>
      normalizeHealthModelResult({
        ...assessment(),
        vetHandoff: null,
      })
    ).toThrow(
      new ServiceUnavailableException('Health screening model returned an invalid assessment')
    );
  });
});

describe('HealthAiService provider privacy boundary', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('uses bearer authentication and store=false without weakening output normalization', async () => {
    const fetchMock = jest.spyOn(global, 'fetch').mockResolvedValue(providerResponse());

    const result = await service().analyze(input);

    expect(result.triage).toBe('monitor');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const request = fetchMock.mock.calls[0]?.[1];
    expect(request?.headers).toMatchObject({
      Authorization: 'Bearer openai-test-key-that-is-long-enough',
      'Content-Type': 'application/json',
    });
    const body = JSON.parse(String(request?.body)) as Record<string, unknown>;
    expect(body.store).toBe(false);
    expect(body.model).toBe('health-test-model');
  });

  it('never reads private provider error bodies into logs or the API error boundary', async () => {
    const privateMarker = 'PRIVATE_HEALTH_OWNER_NOTE=nova-red-patch-at-home';
    const spies = loggerSpies();
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue(new Response(`${privateMarker}; provider-secret-detail`, { status: 503 }));

    await expect(service().analyze({ ...input, concern: privateMarker })).rejects.toThrow(
      'Health screening model is temporarily unavailable'
    );

    expect(spies.warn).toHaveBeenCalledWith(
      'Health model provider failure reason=provider_http_error status=503'
    );
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
    expect(serializedLogCalls(spies)).not.toContain('provider-secret-detail');
  });

  it('classifies malformed provider JSON without logging response content', async () => {
    const privateMarker = 'PRIVATE_MALFORMED_HEALTH_RESPONSE';
    const spies = loggerSpies();
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue(new Response(`{${privateMarker}`, { status: 200 }));

    await expect(service().analyze(input)).rejects.toThrow(
      'Health screening model is temporarily unavailable'
    );

    expect(spies.warn).toHaveBeenCalledWith('Health model provider failure reason=invalid_json');
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
  });

  it('classifies malformed structured output text without logging generated content', async () => {
    const privateMarker = 'PRIVATE_INVALID_STRUCTURED_OUTPUT';
    const spies = loggerSpies();
    jest.spyOn(global, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          output: [{ content: [{ type: 'output_text', text: `{${privateMarker}` }] }],
        }),
        { status: 200 }
      )
    );

    await expect(service().analyze(input)).rejects.toThrow(
      'Health screening model is temporarily unavailable'
    );

    expect(spies.warn).toHaveBeenCalledWith('Health model provider failure reason=invalid_json');
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
  });

  it('classifies AbortError as timeout without logging the exception message', async () => {
    const privateMarker = 'PRIVATE_HEALTH_TIMEOUT_CONTEXT';
    const spies = loggerSpies();
    jest
      .spyOn(global, 'fetch')
      .mockRejectedValue(Object.assign(new Error(privateMarker), { name: 'AbortError' }));

    await expect(service().analyze(input)).rejects.toThrow('Health screening model timed out');

    expect(spies.warn).toHaveBeenCalledWith('Health model provider failure reason=timeout');
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
  });

  it('classifies transport failures without logging arbitrary exception details', async () => {
    const privateMarker = 'PRIVATE_HEALTH_TRANSPORT_DETAIL';
    const spies = loggerSpies();
    jest.spyOn(global, 'fetch').mockRejectedValue(new Error(privateMarker));

    await expect(service().analyze(input)).rejects.toThrow(
      'Health screening model is temporarily unavailable'
    );

    expect(spies.error).toHaveBeenCalledWith(
      'Health model provider failure reason=transport_error'
    );
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
  });

  it('fails closed before network access when the provider is unconfigured', async () => {
    const fetchMock = jest.spyOn(global, 'fetch');

    await expect(service({ OPENAI_API_KEY: undefined }).analyze(input)).rejects.toThrow(
      /not configured/i
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
