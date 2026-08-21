import { ConfigService } from '@nestjs/config';
import { MLService } from './ml.service';
import { LearnedCompatibilityRequest } from '../compatibility/compatibility.types';

const request: LearnedCompatibilityRequest = {
  featureVersion: 'compatibility-features-v1',
  petA: { species: 'dog', behavior: { coverage: 1, energy: 0.5 } },
  petB: { species: 'dog', behavior: { coverage: 1, energy: 0.5 } },
  outcomes: { sampleCount: 0, repeatMeetupCount: 0 },
};

const config = (values: Record<string, string>) =>
  ({ get: (key: string) => values[key] } as unknown as ConfigService);

const learnedScore = (attestationId?: string) => ({
  compatibilityScore: 0.78,
  confidence: 0.82,
  source: 'canonical-test-v1',
  factors: { behaviorCoverage: 1 },
  explanation: ['test'],
  provenance: {
    scorer: 'learned',
    modelVersion: 'canonical-test-v1',
    featureVersion: 'compatibility-features-v1',
    calibrationVersion: 'isotonic-v1',
    generatedAt: new Date().toISOString(),
    fallback: false,
    ...(attestationId
      ? {
          releaseStatus: 'promoted',
          attestationId,
          promotionReceiptSha256: 'a'.repeat(64),
          artifactHashes: {
            modelSha256: 'b'.repeat(64),
            calibrationSha256: 'c'.repeat(64),
            trainingManifestSha256: 'd'.repeat(64),
            featureContractSha256: 'e'.repeat(64),
          },
        }
      : { releaseStatus: 'shadow' }),
  },
});

describe('MLService promoted release trust', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('fails closed to shadow when promoted mode has no pinned attestation', () => {
    const service = new MLService(
      config({
        ML_SERVICE_ENABLED: 'true',
        ML_COMPATIBILITY_MODE: 'promoted',
        ML_SERVICE_URL: 'http://ml.internal',
      }),
    );
    expect(service.getCompatibilityMode()).toBe('shadow');
    expect(service.getStatus().promotedAttestationPinned).toBe(false);
  });

  it('rejects an unattested score in promoted mode', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(learnedScore()), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    const service = new MLService(
      config({
        ML_SERVICE_ENABLED: 'true',
        ML_COMPATIBILITY_MODE: 'promoted',
        ML_PROMOTED_ATTESTATION_ID: 'woof-release-123456',
        ML_SERVICE_URL: 'http://ml.internal',
      }),
    );
    const result = await service.tryPredictCompatibility(request);
    expect(result.score).toBeNull();
    expect(result.fallbackReason).toBe('ml_release_attestation_missing');
  });

  it('rejects a valid promoted envelope when the attestation id is not the pinned release', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(learnedScore('woof-other-release')), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    const service = new MLService(
      config({
        ML_SERVICE_ENABLED: 'true',
        ML_COMPATIBILITY_MODE: 'promoted',
        ML_PROMOTED_ATTESTATION_ID: 'woof-release-123456',
        ML_SERVICE_URL: 'http://ml.internal',
      }),
    );
    const result = await service.tryPredictCompatibility(request);
    expect(result.score).toBeNull();
    expect(result.fallbackReason).toBe('ml_release_attestation_mismatch');
  });

  it('accepts only the exact pinned promoted release envelope', async () => {
    jest.spyOn(global, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(learnedScore('woof-release-123456')), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    const service = new MLService(
      config({
        ML_SERVICE_ENABLED: 'true',
        ML_COMPATIBILITY_MODE: 'promoted',
        ML_PROMOTED_ATTESTATION_ID: 'woof-release-123456',
        ML_SERVICE_URL: 'http://ml.internal',
      }),
    );
    const result = await service.tryPredictCompatibility(request);
    expect(result.score?.provenance.attestationId).toBe('woof-release-123456');
    expect(result.fallbackReason).toBeUndefined();
  });
});
