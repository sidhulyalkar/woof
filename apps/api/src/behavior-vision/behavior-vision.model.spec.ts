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

describe('BehaviorVisionModelService release qualification', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('qualifies an exact pinned release and stamps qualification on the API side', async () => {
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

  it('refuses a configured service when deployment release pinning is incomplete', async () => {
    const fetchMock = jest.spyOn(global, 'fetch');

    await expect(
      service({ BEHAVIOR_VISION_ARTIFACT_SHA256: undefined }).analyze(input)
    ).rejects.toThrow(/qualified model release pin/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
