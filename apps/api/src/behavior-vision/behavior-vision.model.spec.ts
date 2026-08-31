import { Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { BehaviorVisionModelService } from './behavior-vision.model';
import {
  BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
  BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
  type BehaviorVisionModelAnalysis,
} from './behavior-vision.types';

const ARTIFACT_SHA256 = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';

const releaseConfig = {
  BEHAVIOR_VISION_SERVICE_URL: 'https://behavior.example.com',
  BEHAVIOR_VISION_SERVICE_TOKEN: 'behavior-service-token',
  BEHAVIOR_VISION_RELEASE_ID: 'behavior-shadow-2026-08-27',
  BEHAVIOR_VISION_MODEL_VERSION: 'shadow-model-1',
  BEHAVIOR_VISION_FEATURE_VERSION: 'features-1',
  BEHAVIOR_VISION_ARTIFACT_SHA256: ARTIFACT_SHA256,
};

function service(overrides: Record<string, string | undefined> = {}) {
  const values: Record<string, string | undefined> = { ...releaseConfig, ...overrides };
  const config = {
    get: jest.fn((key: string) => values[key]),
  };
  return new BehaviorVisionModelService(config as unknown as ConfigService);
}

function payload(
  overrides: Partial<BehaviorVisionModelAnalysis> = {}
): BehaviorVisionModelAnalysis {
  return {
    schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
    releaseId: releaseConfig.BEHAVIOR_VISION_RELEASE_ID,
    modelVersion: releaseConfig.BEHAVIOR_VISION_MODEL_VERSION,
    featureVersion: releaseConfig.BEHAVIOR_VISION_FEATURE_VERSION,
    artifactSha256: ARTIFACT_SHA256,
    mediaQuality: { usable: true, confidence: 0.9, issues: [], recaptureInstructions: [] },
    evidence: [],
    dimensions: [],
    hypotheses: [],
    observableSummary: 'Observable movement only.',
    uncertainty: 'Internal state cannot be determined from video alone.',
    ...overrides,
  };
}

const input = {
  pet: {
    name: 'Nova',
    species: 'DOG',
    breed: null,
    ageYears: 4,
    temperament: null,
  },
  context: {
    context: 'street' as const,
    phase: 'baseline' as const,
    handlerAction: 'none' as const,
    leashState: 'loose' as const,
    otherDogsPresent: true,
    audioAnalysisAllowed: false,
  },
  media: {
    mimeType: 'video/webm',
    bytes: Buffer.from('transient-private-video'),
    filename: 'nova.webm',
  },
};

function loggerSpies() {
  return {
    warn: jest.spyOn(Logger.prototype, 'warn').mockImplementation(() => undefined),
    error: jest.spyOn(Logger.prototype, 'error').mockImplementation(() => undefined),
  };
}

function serializedLogCalls(spies: ReturnType<typeof loggerSpies>) {
  return JSON.stringify([...spies.warn.mock.calls, ...spies.error.mock.calls]);
}

describe('BehaviorVisionModelService release qualification', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('qualifies an exact pinned release, authenticates upstream, and stamps qualification on the API side', async () => {
    const upstream = payload({
      artifactSha256: ARTIFACT_SHA256.toUpperCase(),
      releaseQualification: {
        qualificationVersion: BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
        qualified: true,
        releaseId: 'untrusted-upstream-claim',
        modelVersion: 'untrusted',
        featureVersion: 'untrusted',
        artifactSha256: 'b'.repeat(64),
        responseContract: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
      },
    });
    const fetchMock = jest
      .spyOn(global, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify(upstream), { status: 200 }));

    const result = await service().analyze(input);

    expect(result.releaseQualification).toEqual({
      qualificationVersion: BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
      qualified: true,
      releaseId: releaseConfig.BEHAVIOR_VISION_RELEASE_ID,
      modelVersion: releaseConfig.BEHAVIOR_VISION_MODEL_VERSION,
      featureVersion: releaseConfig.BEHAVIOR_VISION_FEATURE_VERSION,
      artifactSha256: ARTIFACT_SHA256,
      responseContract: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
    });
    expect(result.releaseId).toBe(releaseConfig.BEHAVIOR_VISION_RELEASE_ID);
    expect(result.artifactSha256).toBe(ARTIFACT_SHA256);

    const request = fetchMock.mock.calls[0]?.[1];
    const form = request?.body as FormData;
    const metadata = JSON.parse(String(form.get('metadata'))) as Record<string, unknown>;
    expect(metadata.expectedRelease).toEqual({
      releaseId: releaseConfig.BEHAVIOR_VISION_RELEASE_ID,
      modelVersion: releaseConfig.BEHAVIOR_VISION_MODEL_VERSION,
      featureVersion: releaseConfig.BEHAVIOR_VISION_FEATURE_VERSION,
      artifactSha256: ARTIFACT_SHA256,
      responseContract: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
    });
    expect(request?.headers).toMatchObject({
      Authorization: `Bearer ${releaseConfig.BEHAVIOR_VISION_SERVICE_TOKEN}`,
    });
  });

  it('rejects a model response from a different artifact before qualification', async () => {
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue(
        new Response(JSON.stringify(payload({ artifactSha256: 'b'.repeat(64) })), { status: 200 })
      );

    await expect(service().analyze(input)).rejects.toThrow(/release identity/i);
  });

  it('rejects model or feature version drift even when the artifact-shaped response is valid', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify(
          payload({
            modelVersion: 'shadow-model-2',
            featureVersion: 'features-2',
          })
        ),
        { status: 200 }
      )
    );

    await expect(service().analyze(input)).rejects.toThrow(/release identity/i);
  });

  it('rejects malformed response contracts without promoting partial provider output', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          ...payload(),
          dimensions: null,
        }),
        { status: 200 }
      )
    );

    await expect(service().analyze(input)).rejects.toThrow(/invalid observation contract/i);
  });

  it('never reads private provider error bodies into logs or the API error boundary', async () => {
    const privateMarker = 'PRIVATE_OWNER_NOTE=nova-reacts-at-elm-street';
    const spies = loggerSpies();
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue(
        new Response(`${privateMarker}; token=secret-provider-detail`, { status: 503 })
      );

    await expect(service().analyze({ ...input, question: privateMarker })).rejects.toThrow(
      'Behavior-video analysis is temporarily unavailable'
    );

    expect(spies.warn).toHaveBeenCalledWith(
      'Behavior vision provider failure reason=provider_http_error status=503'
    );
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
    expect(serializedLogCalls(spies)).not.toContain('secret-provider-detail');
  });

  it('fails closed on invalid JSON without logging provider response content', async () => {
    const privateMarker = 'PRIVATE_INVALID_JSON_OWNER_CONTEXT';
    const spies = loggerSpies();
    jest
      .spyOn(global, 'fetch')
      .mockResolvedValue(new Response(`{${privateMarker}`, { status: 200 }));

    await expect(service().analyze(input)).rejects.toThrow(
      'Behavior-video analysis is temporarily unavailable'
    );

    expect(spies.warn).toHaveBeenCalledWith('Behavior vision provider failure reason=invalid_json');
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
  });

  it('classifies AbortError as a timeout without logging the underlying exception message', async () => {
    const privateMarker = 'PRIVATE_TIMEOUT_REQUEST_STATE';
    const spies = loggerSpies();
    const timeoutError = Object.assign(new Error(privateMarker), { name: 'AbortError' });
    jest.spyOn(global, 'fetch').mockRejectedValue(timeoutError);

    await expect(service().analyze(input)).rejects.toThrow('Behavior-video analysis timed out');

    expect(spies.warn).toHaveBeenCalledWith('Behavior vision provider failure reason=timeout');
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
  });

  it('classifies transport failures without logging arbitrary fetch exception details', async () => {
    const privateMarker = 'PRIVATE_TRANSPORT_DETAIL_WITH_REQUEST_METADATA';
    const spies = loggerSpies();
    jest.spyOn(global, 'fetch').mockRejectedValue(new Error(privateMarker));

    await expect(service().analyze(input)).rejects.toThrow(
      'Behavior-video analysis is temporarily unavailable'
    );

    expect(spies.error).toHaveBeenCalledWith(
      'Behavior vision provider failure reason=transport_error'
    );
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
  });

  it('removes audio-derived evidence when audio analysis is disabled', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify(
          payload({
            evidence: [
              { label: 'bark', source: 'audio', confidence: 0.9 },
              { label: 'weight shift', source: 'pose', confidence: 0.8 },
            ],
            dimensions: [
              {
                dimension: 'arousal',
                value: 0.7,
                confidence: 0.8,
                basis: ['audio bark', 'motion'],
              },
              {
                dimension: 'body-tension',
                value: 0.4,
                confidence: 0.7,
                basis: ['pose weight shift'],
              },
            ],
          })
        ),
        { status: 200 }
      )
    );

    const result = await service().analyze(input);

    expect(result.evidence).toEqual([{ label: 'weight shift', source: 'pose', confidence: 0.8 }]);
    expect(result.dimensions).toEqual([
      {
        dimension: 'body-tension',
        value: 0.4,
        confidence: 0.7,
        basis: ['pose weight shift'],
      },
    ]);
  });

  it('returns an explicit unavailable boundary without making a request when no service is configured', async () => {
    const fetchMock = jest.spyOn(global, 'fetch');

    await expect(
      service({ BEHAVIOR_VISION_SERVICE_URL: undefined }).analyze(input)
    ).rejects.toThrow(/not configured/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('refuses a configured service when deployment release pinning is incomplete', async () => {
    const fetchMock = jest.spyOn(global, 'fetch');

    await expect(
      service({ BEHAVIOR_VISION_ARTIFACT_SHA256: undefined }).analyze(input)
    ).rejects.toThrow(/qualified model release pin/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
