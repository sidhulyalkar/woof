import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
  BEHAVIOR_DIMENSIONS,
  BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
  type BehaviorDimension,
  type BehaviorObservationContext,
  type BehaviorVisionModelAnalysis,
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

@Injectable()
export class BehaviorVisionModelService {
  private readonly logger = new Logger(BehaviorVisionModelService.name);
  private readonly serviceUrl: string | null;
  private readonly serviceToken: string | null;
  private readonly timeoutMs: number;

  constructor(private readonly config: ConfigService) {
    this.serviceUrl = this.config.get<string>('BEHAVIOR_VISION_SERVICE_URL') || null;
    this.serviceToken = this.config.get<string>('BEHAVIOR_VISION_SERVICE_TOKEN') || null;
    this.timeoutMs = Math.max(
      5000,
      Math.min(90000, Number(this.config.get('BEHAVIOR_VISION_TIMEOUT_MS') || 45000))
    );
  }

  isConfigured() {
    return Boolean(this.serviceUrl);
  }

  async analyze(input: BehaviorVisionModelInput): Promise<BehaviorVisionModelAnalysis> {
    if (!this.serviceUrl) {
      throw new ServiceUnavailableException(
        'Specialized behavior-video analysis is not configured in this environment'
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
        const body = await response.text();
        this.logger.warn(
          `Behavior vision service failed with ${response.status}: ${body.slice(0, 500)}`
        );
        throw new ServiceUnavailableException(
          'Behavior-video analysis is temporarily unavailable'
        );
      }

      const payload = (await response.json()) as BehaviorVisionModelAnalysis;
      return this.validate(payload, audioAllowed);
    } catch (error) {
      if (error instanceof ServiceUnavailableException) throw error;
      if (error instanceof Error && error.name === 'AbortError') {
        throw new ServiceUnavailableException('Behavior-video analysis timed out');
      }
      this.logger.error(
        `Behavior vision analysis failed: ${error instanceof Error ? error.message : 'unknown error'}`
      );
      throw new ServiceUnavailableException('Behavior-video analysis is temporarily unavailable');
    } finally {
      clearTimeout(timeout);
    }
  }

  private validate(
    result: BehaviorVisionModelAnalysis,
    audioAllowed: boolean
  ): BehaviorVisionModelAnalysis {
    if (
      !result ||
      result.schemaVersion !== BEHAVIOR_OBSERVATION_SCHEMA_VERSION ||
      typeof result.modelVersion !== 'string' ||
      typeof result.featureVersion !== 'string' ||
      typeof result.observableSummary !== 'string' ||
      !result.mediaQuality ||
      !Array.isArray(result.dimensions) ||
      !Array.isArray(result.evidence) ||
      !Array.isArray(result.hypotheses)
    ) {
      throw new ServiceUnavailableException(
        'Behavior-video model returned an invalid observation contract'
      );
    }

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
          ? entry.basis.filter((basis) => audioAllowed || !basis.toLowerCase().includes('audio')).slice(0, 8)
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

  private clamp01(value: number) {
    return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  }
}