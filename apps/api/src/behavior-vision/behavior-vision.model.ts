import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
  BEHAVIOR_DIMENSIONS,
  BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
  BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
  type BehaviorDimension,
  type BehaviorObservationContext,
  type BehaviorVisionModelAnalysis,
  type BehaviorVisionReleaseQualification,
} from './behavior-vision.types';

export type BehaviorVisionModelInput = {
  pet: {
    name: string;
    species: string;
    breed: string | null;
    ageYears: number | null;
    temperament: unknown;
  };
  context: BehaviorObservationContext;
  question?: string;
  priorProfileSummary?: {
    sampleCount: number;
    personalizationConfidence: number;
    baselines: Array<{ dimension: string; mean: number; confidence: number }>;
  };
  media: {
    mimeType: string;
    bytes: Buffer;
    filename: string;
  };
};

type ActiveReleasePin = {
  releaseId: string;
  modelVersion: string;
  featureVersion: string;
  artifactSha256: string;
};

type BehaviorVisionFailureReason =
  'provider_http_error' | 'invalid_json' | 'timeout' | 'transport_error';

@Injectable()
export class BehaviorVisionModelService {
  private readonly logger = new Logger(BehaviorVisionModelService.name);
  private readonly serviceUrl: string | null;
  private readonly serviceToken: string | null;
  private readonly timeoutMs: number;
  private readonly activeRelease: ActiveReleasePin | null;

  constructor(private readonly config: ConfigService) {
    this.serviceUrl = this.config.get<string>('BEHAVIOR_VISION_SERVICE_URL') || null;
    this.serviceToken = this.config.get<string>('BEHAVIOR_VISION_SERVICE_TOKEN') || null;
    this.timeoutMs = Math.max(
      5000,
      Math.min(90000, Number(this.config.get('BEHAVIOR_VISION_TIMEOUT_MS') || 45000))
    );

    const releaseId = this.config.get<string>('BEHAVIOR_VISION_RELEASE_ID')?.trim() || null;
    const modelVersion = this.config.get<string>('BEHAVIOR_VISION_MODEL_VERSION')?.trim() || null;
    const featureVersion =
      this.config.get<string>('BEHAVIOR_VISION_FEATURE_VERSION')?.trim() || null;
    const artifactSha256 =
      this.config.get<string>('BEHAVIOR_VISION_ARTIFACT_SHA256')?.trim().toLowerCase() || null;

    this.activeRelease =
      this.serviceUrl && releaseId && modelVersion && featureVersion && artifactSha256
        ? { releaseId, modelVersion, featureVersion, artifactSha256 }
        : null;
  }

  isConfigured() {
    return Boolean(this.serviceUrl);
  }

  activeReleaseQualification(): BehaviorVisionReleaseQualification | null {
    if (!this.activeRelease || !this.isSha256(this.activeRelease.artifactSha256)) return null;
    return this.qualifyRelease(this.activeRelease);
  }

  async analyze(input: BehaviorVisionModelInput): Promise<BehaviorVisionModelAnalysis> {
    if (!this.serviceUrl) {
      throw new ServiceUnavailableException(
        'Specialized behavior-video analysis is not configured in this environment'
      );
    }
    if (!this.activeRelease || !this.isSha256(this.activeRelease.artifactSha256)) {
      throw new ServiceUnavailableException(
        'Behavior-video analysis is configured without a qualified model release pin'
      );
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    const audioAllowed = input.context.audioAnalysisAllowed === true;

    try {
      const form = new FormData();
      form.append(
        'metadata',
        JSON.stringify({
          schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
          expectedRelease: {
            ...this.activeRelease,
            responseContract: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
          },
          pet: input.pet,
          context: input.context,
          question: input.question ?? null,
          priorProfileSummary: input.priorProfileSummary ?? null,
          policy: {
            objectiveObservationOnly: true,
            noDefinitiveEmotionInference: true,
            noAutomaticGreetingRecommendation: true,
            noHumanFaceRecognition: true,
            noBiometricIdentityInference: true,
            audioAnalysisAllowed: audioAllowed,
          },
        })
      );
      form.append(
        'media',
        new Blob([new Uint8Array(input.media.bytes)], { type: input.media.mimeType }),
        input.media.filename
      );

      const headers: Record<string, string> = {};
      if (this.serviceToken) headers.Authorization = `Bearer ${this.serviceToken}`;

      const response = await fetch(`${this.serviceUrl.replace(/\/$/, '')}/v1/behavior/analyze`, {
        method: 'POST',
        headers,
        body: form,
        signal: controller.signal,
      });

      if (!response.ok) {
        this.warnFailure('provider_http_error', response.status);
        throw new ServiceUnavailableException('Behavior-video analysis is temporarily unavailable');
      }

      let payload: BehaviorVisionModelAnalysis;
      try {
        payload = (await response.json()) as BehaviorVisionModelAnalysis;
      } catch {
        this.warnFailure('invalid_json');
        throw new ServiceUnavailableException('Behavior-video analysis is temporarily unavailable');
      }

      return this.validate(payload, audioAllowed, this.activeRelease);
    } catch (error) {
      if (error instanceof ServiceUnavailableException) throw error;
      if (error instanceof Error && error.name === 'AbortError') {
        this.warnFailure('timeout');
        throw new ServiceUnavailableException('Behavior-video analysis timed out');
      }
      this.errorFailure('transport_error');
      throw new ServiceUnavailableException('Behavior-video analysis is temporarily unavailable');
    } finally {
      clearTimeout(timeout);
    }
  }

  private validate(
    result: BehaviorVisionModelAnalysis,
    audioAllowed: boolean,
    expectedRelease: ActiveReleasePin
  ): BehaviorVisionModelAnalysis {
    if (
      !result ||
      result.schemaVersion !== BEHAVIOR_OBSERVATION_SCHEMA_VERSION ||
      typeof result.modelVersion !== 'string' ||
      typeof result.featureVersion !== 'string' ||
      typeof result.releaseId !== 'string' ||
      typeof result.artifactSha256 !== 'string' ||
      typeof result.observableSummary !== 'string' ||
      typeof result.uncertainty !== 'string' ||
      !result.mediaQuality ||
      !Array.isArray(result.dimensions) ||
      !Array.isArray(result.evidence) ||
      !Array.isArray(result.hypotheses)
    ) {
      throw new ServiceUnavailableException(
        'Behavior-video model returned an invalid observation contract'
      );
    }

    const returnedArtifactSha256 = result.artifactSha256.toLowerCase();
    if (
      result.releaseId !== expectedRelease.releaseId ||
      result.modelVersion !== expectedRelease.modelVersion ||
      result.featureVersion !== expectedRelease.featureVersion ||
      returnedArtifactSha256 !== expectedRelease.artifactSha256 ||
      !this.isSha256(returnedArtifactSha256)
    ) {
      throw new ServiceUnavailableException(
        'Behavior-video model release identity does not match the qualified deployment pin'
      );
    }

    const releaseQualification = this.qualifyRelease(expectedRelease);

    const allowedDimensions = new Set<BehaviorDimension>(BEHAVIOR_DIMENSIONS);
    const dimensions = result.dimensions
      .filter((entry) => allowedDimensions.has(entry.dimension))
      .filter(
        (entry) =>
          audioAllowed ||
          !Array.isArray(entry.basis) ||
          !entry.basis.some((basis) => basis.toLowerCase().includes('audio'))
      )
      .map((entry) => ({
        ...entry,
        value: this.clamp01(entry.value),
        confidence: this.clamp01(entry.confidence),
        basis: Array.isArray(entry.basis)
          ? entry.basis
              .filter((basis) => audioAllowed || !basis.toLowerCase().includes('audio'))
              .slice(0, 8)
          : [],
      }));

    const evidence = result.evidence
      .filter((entry) => audioAllowed || entry.source !== 'audio')
      .slice(0, 40)
      .map((entry) => ({
        ...entry,
        confidence: this.clamp01(entry.confidence),
      }));

    return {
      ...result,
      releaseId: expectedRelease.releaseId,
      artifactSha256: expectedRelease.artifactSha256,
      releaseQualification,
      mediaQuality: {
        usable: Boolean(result.mediaQuality.usable),
        confidence: this.clamp01(result.mediaQuality.confidence),
        issues: Array.isArray(result.mediaQuality.issues)
          ? result.mediaQuality.issues.slice(0, 8)
          : [],
        recaptureInstructions: Array.isArray(result.mediaQuality.recaptureInstructions)
          ? result.mediaQuality.recaptureInstructions.slice(0, 6)
          : [],
      },
      evidence,
      dimensions,
      hypotheses: result.hypotheses.slice(0, 6).map((entry) => ({
        ...entry,
        confidence: this.clamp01(entry.confidence),
        supportingEvidence: Array.isArray(entry.supportingEvidence)
          ? entry.supportingEvidence.slice(0, 8)
          : [],
        contradictoryEvidence: Array.isArray(entry.contradictoryEvidence)
          ? entry.contradictoryEvidence.slice(0, 8)
          : [],
      })),
    };
  }

  private warnFailure(reason: BehaviorVisionFailureReason, status?: number) {
    const statusSuffix = status === undefined ? '' : ` status=${status}`;
    this.logger.warn(`Behavior vision provider failure reason=${reason}${statusSuffix}`);
  }

  private errorFailure(reason: BehaviorVisionFailureReason) {
    this.logger.error(`Behavior vision provider failure reason=${reason}`);
  }

  private qualifyRelease(release: ActiveReleasePin): BehaviorVisionReleaseQualification {
    return {
      qualificationVersion: BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
      qualified: true,
      ...release,
      responseContract: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
    };
  }

  private isSha256(value: string) {
    return /^[a-f0-9]{64}$/.test(value);
  }

  private clamp01(value: number) {
    return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  }
}
